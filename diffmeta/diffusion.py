import math
from functools import partial
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from einops import reduce

from tqdm.auto import tqdm
from pytorch_msssim import SSIM

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class CustomDataParallel(nn.Module):
    def __init__(self, model, device_ids):
        super(CustomDataParallel, self).__init__()
        self.model = nn.DataParallel(model, device_ids=device_ids).cuda()

    def forward(self, *input):
        return self.model(*input)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        image_size,
        timesteps = 1000,
        sampling_timesteps = 10,
        loss_type = 'l2',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1,
        train=True

    ):
        super().__init__()

        self.train=train
        if self.train:
            print("Train Mode")
        else:
            print("Test Mode")


        self.model = model

        self.image_size = (image_size, image_size)

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training
        self.time_pairs = self.set_timepairs()
        assert self.sampling_timesteps <= timesteps
        # self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond, clip_x_start = False):#false
        model_output = self.model(x, t, x_self_cond)
        predict_noise, predict_p_thick = model_output
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = predict_noise
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = predict_noise
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = predict_noise
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), predict_p_thick

    def p_mean_variance(self, x, t, x_self_cond , clip_denoised = True):
        preds, predict_p_thick = self.model_predictions(x, t, x_self_cond)
        # print('predict_p_thick', predict_p_thick)
        x_start = preds.pred_x_start
        # print('x_start1', x_start)
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        # print('x_start2', x_start)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start, predict_p_thick

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start, predict_p_thick = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start, predict_p_thick

    @torch.no_grad()
    def p_sample_loop(self, shape,self_cond):
        batch, device = shape[0], self.betas.device
        
        img = torch.randn(shape, device=device)

        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # self_cond = x_start if self.self_condition else None
            img, x_start, predict_p_thick = self.p_sample(img, t, self_cond)
        # img = unnormalize_to_zero_to_one(img)
        # print('predict_p_thick', predict_p_thick)
        return img, predict_p_thick
    
    def calculate_log_probs(self, prev_sample, prev_sample_mean, std_dev_t):
        std_dev_t = torch.clip(std_dev_t, 1e-6)
        log_probs = -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_dev_t ** 2) - torch.log(std_dev_t) - math.log(math.sqrt(2 * math.pi))
        return log_probs.mean(dim=tuple(range(1, prev_sample_mean.ndim)))

    def set_timepairs(self):
        total_timesteps, sampling_timesteps = self.num_timesteps, self.sampling_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        return time_pairs
    
    def calculate_loss(self, lentents, next_latents, self_cond, time, time_next, eta=1.0, clip_denoised = True):
        batch, device = len(lentents), self.betas.device
        time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
        pred_noise, x_start, *_ = self.model_predictions(lentents, time_cond, self_cond, clip_x_start = clip_denoised)
        alpha = self.alphas_cumprod[time]
        if time_next < 0:
            alpha_next = self.alphas_cumprod[0]
        else:
            alpha_next = self.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(lentents)
        prev_sample_mean = x_start * alpha_next.sqrt() + c * pred_noise
        std_dev_t = sigma
        log_prob = self.calculate_log_probs(next_latents, prev_sample_mean, std_dev_t)
        # print("log", log_prob)
        return log_prob

    @torch.no_grad()
    def ddim_sample(self, shape, self_cond, eta=1.0, clip_denoised = True):
        batch, device, time_pairs = shape[0], self.betas.device, self.time_pairs
        img = torch.randn(shape, device = device)

        x_start = None
        all_step_preds, log_probs = [img], []

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            # self_cond = x_start if self.self_condition else None
            preds, predict_p_thick = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)
            x_start = preds.pred_x_start
            pred_noise = preds.pred_noise
            alpha = self.alphas_cumprod[time]
            if time_next < 0:
                alpha_next = self.alphas_cumprod[0]
            else:
                alpha_next = self.alphas_cumprod[time_next]
            
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)
            prev_sample_mean = x_start * alpha_next.sqrt() + c * pred_noise
            img = prev_sample_mean + sigma * noise
            prev_sample = img
            std_dev_t = sigma
            log_prob = self.calculate_log_probs(prev_sample, prev_sample_mean, std_dev_t)
            # print("log", log_prob)
            log_probs.append(log_prob)
            all_step_preds.append(img)
        # return img, torch.stack(all_step_preds), torch.stack(log_probs)]
        return img, predict_p_thick

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss

        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss

        elif self.loss_type=="huber_loss":
            return F.huber_loss
        elif self.loss_type=="ssim":
            return SSIM(data_range=1, size_average=True, channel=1)
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
    def loss_calculation(self,model_out, target, p_thick, t, reduction = 'none'):
        predict_noise, predict_p_thick = model_out
        noise_loss = self.loss_fn(predict_noise, target, reduction=reduction)
        noise_loss = reduce(noise_loss, 'b ... -> b (...)', 'mean')
        noise_loss = noise_loss * extract(self.p2_loss_weight, t, noise_loss.shape)
        # return noise_loss.mean()
        p_thick_loss = self.loss_fn(predict_p_thick, p_thick, reduction=reduction)
        return noise_loss.mean() + 0.1*p_thick_loss.mean()
        
    def p_losses(self, x_start, t, x_self_cond, p_thick):
        b, c, h, w = x_start.shape
        noise = torch.randn_like(x_start)

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        # x_self_cond = None
        # if self.self_condition and random() < 0.5:
        #     with torch.no_grad():
        #         x_self_cond = self.model_predictions(x, t).pred_x_start
        #         x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        loss = self.loss_calculation(model_out, target, p_thick, t, reduction = 'none')
        return loss.mean()

    def forward(self, img, x_self_cond, p_thick=None, next_latents=None, time=None, time_next=None, train=None, is_ddim_sampling=False):
        train = self.train if train==None else train
        if train:
            b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
            # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            return self.p_losses(img, t, x_self_cond, p_thick)
        
        if not train:
            if next_latents:
                return self.calculate_loss(img, next_latents, x_self_cond, time, time_next)
            else:      
                if not is_ddim_sampling:
                    print("DDPM sampling")
                    return self.p_sample_loop(img.shape, x_self_cond)

                else:
                    print("DDIM sampling")
                    return self.ddim_sample(img.shape, x_self_cond)
    
