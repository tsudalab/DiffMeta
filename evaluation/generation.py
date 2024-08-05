import numpy as np
import torch
import argparse
import os
from PIL import Image
import cv2
from DiffMeta.diffmeta.diffusion import GaussianDiffusion
from DiffMeta.diffmeta.unet import Unet
from DiffMeta.benchmark.models import cVAE_GSNN, cVAE_hybrid, cGAN, SimulatorNet_new_fc

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def shape_to_image(shape, col=255):
    """Convert shape data to an image."""
    shape = inverse_transform(shape)
    shape = shape[0].to('cpu') * col
    img = Image.fromarray(np.uint8(shape.detach().numpy()))
    return shape, img

def draw_image(diff, vae, gan, idx):
    """Draw and save images for different models."""
    images = {}
    if diff is not None:
        diff_shape, diff_img = shape_to_image(diff)
        images['diff'] = (diff_shape, diff_img)
    if vae is not None:
        vae_shape, vae_img = shape_to_image(vae)
        images['vae'] = (vae_shape, vae_img)
    if gan is not None:
        gan_shape, gan_img = shape_to_image(gan)
        images['gan'] = (gan_shape, gan_img)
   
    if images:
        width, height = next(iter(images.values()))[1].size
        new_image = Image.new("RGB", (width * len(images), height))
        for i, (key, (_, img)) in enumerate(images.items()):
            new_image.paste(img, (width * i, 0))
        new_image.save(f"images/target_images/predict_image{idx}.png")
    
    return images

def inverse_transform(x):
    """Inverse transform for shape data."""
    x = (x + 1) / 2
    x = (x.clamp(-1, 1) + 1) / 2
    x[x < 0.5] = 0
    x[x >= 0.5] = 1
    return x

@torch.no_grad()
def forward_model_simulator(forward_model, shape, p_thick):
    """Generate spectrum prediction using the forward model."""
    with torch.no_grad():
        spectrum_pred = forward_model(shape, p_thick)
    return spectrum_pred

@torch.no_grad()
def gan_generator(gan_model, spectrums):
    """Generate predictions using the GAN model."""
    z = gan_model.sample_noise(len(spectrums), 1).to(DEVICE)
    img_pred, p_thick_pred = gan_model.Generator(spectrums, z)
    p_thick_pred = process_predicted_p_thick(p_thick_pred)
    return img_pred, p_thick_pred

@torch.no_grad()
def vae_generator(vae_model, spectrums):
    """Generate predictions using the VAE model."""
    img_pred, p_thick_pred, _, _, _, _ = vae_model.inference(spectrums)
    p_thick_pred = process_predicted_p_thick(p_thick_pred)
    return img_pred, p_thick_pred

@torch.no_grad()
def diff_generator(diff_model, spectrums):
    """Generate predictions using the Diffusion model."""
    inputs = torch.randn((len(spectrums), 1, 64, 64), device=DEVICE)
    targets = spectrums.unsqueeze(1)
    predict_shapes, predict_p_thick = diff_model(inputs, targets)
    predict_p_thick = process_predicted_p_thick(predict_p_thick)
    return predict_shapes, predict_p_thick

def criterion_shape(origin_shape, predicted_shape):
    """Calculate similarity between original and predicted shapes using shape matching."""
    origin_shape, predicted_shape = inverse_transform(origin_shape), inverse_transform(predicted_shape)
    origin_shape, predicted_shape = origin_shape.cpu().detach().numpy(), predicted_shape.cpu().detach().numpy()
    scores = []
    for i in range(len(origin_shape)):
        similarity = cv2.matchShapes(origin_shape[i][0] * 255, predicted_shape[i][0] * 255, 2, 0.0)
        scores.append(similarity)
    return scores

def process_predicted_p_thick(predict_p_thick):
    """Process and recover predicted thicknesses from normalized values."""
    original_mins = [3, 0, 0, 0]
    original_maxs = [8, 0.8, 1, 0.2]
    predict_p_thick = predict_p_thick.cpu().detach().numpy()
    recovered_p_thick = np.column_stack([
        inverse_min_max_normalize(predict_p_thick[:, i], original_mins[i], original_maxs[i])
        for i in range(predict_p_thick.shape[1])
    ])
    return recovered_p_thick

def inverse_min_max_normalize(norm_data, original_min, original_max):
    """Inverse min-max normalization."""
    return norm_data * (original_max - original_min) + original_min

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate predictions using selected models.")
    parser.add_argument('filename', type=str, help='Path to the target spectrum file')
    parser.add_argument('--models', nargs='+', choices=['diff', 'vae', 'gan'], help='Models to use for predictions', required=True)
    args = parser.parse_args()
    
    DEVICE = 'cuda'
    forward_model_path = '../models/simulator.pth'
    vae_path = '../models/vaemeta.pth'
    gan_path = '../models/ganmeta.pth'
    diff_path = '../models/diffmeta.pth'

    # Load forward model
    forward_model = SimulatorNet_new_fc(spec_dim=400, d=64).to(DEVICE)
    forward_model.load_state_dict(torch.load(forward_model_path)['model_state_dict'])
  
    # Load GAN model
    if 'gan' in args.models:
        print('Loading GAN model...')
        configs = torch.load(gan_path, map_location=DEVICE)['configs']
        gan_model = cGAN(img_size=64, spec_dim=configs.spec_dim, noise_dim=configs.noise_dim, d=configs.net_depth, thickness=configs.layers, k_size=configs.k_size, k_pad=configs.k_pad).to(DEVICE)
        gan_model.load_state_dict(torch.load(gan_path, map_location=DEVICE)['model_state_dict'])
    
    # Load VAE model
    if 'vae' in args.models:
        print('Loading VAE model...')
        configs = torch.load(vae_path, map_location=DEVICE)['configs']
        vae_gsnn = cVAE_GSNN(spec_dim=configs.spec_dim, latent_dim=configs.latent_dim, d=configs.net_depth, thickness=configs.layers, k_size=configs.k_size, k_pad=configs.k_pad).to(DEVICE)
        vae_model = cVAE_hybrid(forward_model, vae_gsnn).to(DEVICE)
        vae_model.load_state_dict(torch.load(vae_path, map_location=DEVICE)['model_state_dict'])
    
    # Load Diffusion model
    if 'diff' in args.models:
        print('Loading Diffusion model...')
        diff_states = torch.load(diff_path, map_location=DEVICE)
        diff_model = GaussianDiffusion(Unet(dim=128, device=DEVICE, dim_mults=(1, 1, 2, 2, 4)), image_size=64, p2_loss_weight_k=0, timesteps=1000, loss_type="l2", train=False).to(DEVICE)
        diff_model.load_state_dict(diff_states["model_state_dict"])
        pytorch_total_params = sum(p.numel() for p in diff_model.parameters())
        print(diff_states['epoch'], pytorch_total_params)

    # Load target spectrum
    target = np.load(args.filename)
    spectrums = torch.tensor(target, device=DEVICE).float()
    
    with torch.no_grad():
        spectrums = spectrums.view(1, 400)
        print(spectrums.size())
        for j in range(10):
            torch.manual_seed(j)
            diff_prediction = vae_prediction = gan_prediction = None
            diff_predict_p_thick = vae_predict_p_thick = gan_predict_p_thick = None

            if 'diff' in args.models:
                diff_prediction, diff_predict_p_thick = diff_generator(diff_model, spectrums)
            if 'vae' in args.models:
                vae_prediction, vae_predict_p_thick = vae_generator(vae_model, spectrums)
            if 'gan' in args.models:
                gan_prediction, gan_predict_p_thick = gan_generator(gan_model, spectrums)

            # Draw images and save results
            images = draw_image(diff_prediction[0] if diff_prediction is not None else None, 
                                vae_prediction[0] if vae_prediction is not None else None, 
                                gan_prediction[0] if gan_prediction is not None else None, j)
            
            if 'diff' in args.models:
                np.savez(f'diff_target_shape_{j}.npz', images=images['diff'][0].detach().numpy(), p_thick=diff_predict_p_thick)
            if 'vae' in args.models:
                np.savez(f'vea_target_shape_{j}.npz', images=images['vae'][0].detach().numpy(), p_thick=vae_predict_p_thick)
            if 'gan' in args.models:
                np.savez(f'gan_target_shape_{j}.npz', images=images['gan'][0].detach().numpy(), p_thick=gan_predict_p_thick)
