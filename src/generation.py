import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dataset import MetaMaterials
from torch.utils.data import DataLoader
from diffusion import *
from net_b import *
import os
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"]='0'
from fastprogress import progress_bar
import cv2
from DiffMeta.benchmark.models import cVAE_GSNN, cVAE_hybrid, cGAN, SimulatorNet_new_fc
from sklearn.metrics import mean_squared_error as mse
def spectrum_prediction(origin_spec, diff_spec, vae_spec, gan_spec, idx):
    origin_color = '#566CA5'
    diff_color = '#D2352C'
    vae_color = '#68AC57'
    gan_color = '#8E549E'
    origin_spec, diff_spec, vae_spec, gan_spec = origin_spec.cpu().detach().numpy(), diff_spec.cpu().detach().numpy(), vae_spec.cpu().detach().numpy(), gan_spec.cpu().detach().numpy()
    wave = np.linspace(3, 15, 400)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(wave, origin_spec, linewidth=1.8, linestyle = 'solid', label = 'Real TE', c=origin_color)
    ax.plot(wave, diff_spec,  linewidth=1.8, linestyle = 'dashed', label = 'DMRL', c=diff_color)
    ax.plot(wave, vae_spec,  linewidth=1.8, linestyle = 'dashed', label = 'C-VAE', c=vae_color)
    ax.plot(wave, gan_spec,  linewidth=1.8, linestyle = 'dashed', label = 'C-GAN', c=gan_color)
    ax.set_ylim([0.0, 1.0])
    ax.set_ylabel('Emissivity')
    ax.set_xlabel('Wavelength (um)')
    ax.legend(loc = 'upper left')  
    plt.tight_layout()
    plt.savefig('images/spectrum' + str(idx) + '.png')
    plt.clf()
    return np.sqrt(mse(diff_spec, origin_spec)), np.sqrt(mse(vae_spec, origin_spec)), np.sqrt(mse(gan_spec, origin_spec))

def shape_to_image(shape, col=255):
    shape = inverse_transform(shape)
    shape = shape[0].to('cpu') * col
    img = Image.fromarray(np.uint8(shape.detach().numpy()))
    return shape, img

def draw_image(origin, diff, vae, gan, idx):
    origin_shape, origin_img = shape_to_image(origin)
    diff_shape, diff_img = shape_to_image(diff)
    vae_shape, vae_img = shape_to_image(vae)
    gan_shape, gan_img = shape_to_image(gan)
   
    width, height = origin_img.size
    new_image = Image.new("RGB", (width*4, height))

    new_image.paste(origin_img, (0, 0))
    new_image.paste(diff_img, (width, 0))
    new_image.paste(vae_img, (width*2, 0))
    new_image.paste(gan_img, (width*3, 0))
    new_image.save("images/target_images/predict_image"+str(idx)+".png")
    return origin_shape, diff_shape, vae_shape, gan_shape

def inverse_transform(x):
    x = (x+1) / 2
    x = (x.clamp(-1, 1) + 1) / 2
    x[x < 0.5] = 0
    x[x >= 0.5] = 1
    return x

@torch.no_grad()
def forward_model_simulator(forward_model, shape, p_thick):  
    with torch.no_grad():
        spectrum_pred = forward_model(shape, p_thick)
    return spectrum_pred

@torch.no_grad()
def gan_generator(gan_model, spectrums):
    z = gan_model.sample_noise(len(spectrums), 1).to(DEVICE)
    img_pred, p_thick_pred = gan_model.Generator(spectrums, z)
    p_thick_pred = process_predicted_p_thick(p_thick_pred)
    return img_pred, p_thick_pred

@torch.no_grad()
def vae_generator(vae_model, spectrums):
    img_pred, p_thick_pred, _, _, _, _ = vae_model.inference(spectrums)
    p_thick_pred = process_predicted_p_thick(p_thick_pred)
    return img_pred, p_thick_pred

@torch.no_grad()
def diff_generator(diff_model, spectrums):
    inputs=torch.randn((len(spectrums), 1, 64, 64), device=DEVICE)
    targets = spectrums.unsqueeze(1)
    predict_shapes, predict_p_thick = diff_model(inputs, targets)
    predict_p_thick = process_predicted_p_thick(predict_p_thick)
    return predict_shapes, predict_p_thick

def criterion_shape(origin_shape, predicted_shape):
    origin_shape, predicted_shape = inverse_transform(origin_shape), inverse_transform(predicted_shape)
    origin_shape, predicted_shape = origin_shape.cpu().detach().numpy(), predicted_shape.cpu().detach().numpy()
    scores = []
    for i in range(len(origin_shape)):
        simi = cv2.matchShapes(origin_shape[i][0]*255, predicted_shape[i][0]*255, 2, 0.0)
        scores.append(simi)
    return scores

def process_predicted_p_thick(predict_p_thick):
    original_mins = [3, 0, 0, 0]  
    original_maxs = [8, 0.8, 1, 0.2]
    predict_p_thick = predict_p_thick.cpu().detach().numpy()
    recovered_p_thick = np.column_stack([
        inverse_min_max_normalize(predict_p_thick[:, i], original_mins[i], original_maxs[i])
        for i in range(predict_p_thick.shape[1])
    ])
    return recovered_p_thick

def inverse_min_max_normalize(norm_data, original_min, original_max):
    return norm_data * (original_max - original_min) + original_min

if __name__ == '__main__':
    batch_size = 64
    DEVICE = 'cuda'
    test_dataset = MetaMaterials('test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    pbar = progress_bar(test_loader, leave=False)
    forword_path =  '../models/simulator.pth'
    vae_path =  '../models/vaemeta.pth'
    gan_path =  '../models/ganmeta.pth'
    diff_path =  '../models/diffmeta.pth'
    
    print('loading spectrum predictor ...')
    forward_model = SimulatorNet_new_fc(spec_dim=400, d=64).to(DEVICE)
    forward_model_dict = torch.load(forword_path)['model_state_dict']
    forward_model.load_state_dict(forward_model_dict)
  
    print('loading gan model ...')
    configs = torch.load(gan_path, map_location=DEVICE)['configs']
    gan_model = cGAN(img_size=64, spec_dim=configs.spec_dim, noise_dim=configs.noise_dim, d=configs.net_depth, thickness=configs.layers, k_size=configs.k_size, k_pad=configs.k_pad).to(DEVICE)
    gan_model.load_state_dict(torch.load(gan_path, map_location=DEVICE)['model_state_dict'])
    
    print('loading vae model ...')
    configs = torch.load(vae_path, map_location=DEVICE)['configs']
    vae_gsnn = cVAE_GSNN(spec_dim=configs.spec_dim, latent_dim=configs.latent_dim, d=configs.net_depth, thickness=configs.layers, k_size=configs.k_size, k_pad=configs.k_pad).to(DEVICE)
    vae_model = cVAE_hybrid(forward_model, vae_gsnn).to(DEVICE)
    vae_model.load_state_dict(torch.load(vae_path, map_location=DEVICE)['model_state_dict'])
    
    print('loading diff model ...')
    diff_states=torch.load(diff_path, map_location=DEVICE)
    diff_model=GaussianDiffusion(Unet(dim=128,device=DEVICE, dim_mults=(1,1,2,2,4)),image_size=64,p2_loss_weight_k=0,timesteps=1000,loss_type="l2", train=False).to(DEVICE)
   
    diff_model.load_state_dict(diff_states["model_state_dict"])
    pytorch_total_params = sum(p.numel() for p in diff_model.parameters())
    print(diff_states['epoch'], pytorch_total_params)

    criterion = nn.MSELoss()
    with torch.no_grad():
        for shapes, spectrums, p_thick in pbar:
            shapes, spectrums, p_thick = shapes.to(DEVICE), spectrums.to(DEVICE), p_thick.to(DEVICE)
            
            forward_spec = forward_model_simulator(forward_model, shapes, p_thick) 
            diff_prediction, diff_predict_p_thick = diff_generator(diff_model, spectrums)
            vae_prediction, vae_predict_p_thick = vae_generator(vae_gsnn, spectrums)
            gan_prediction, gan_predict_p_thick = gan_generator(gan_model, spectrums)
            print('shape rmse: diff', np.sqrt(criterion(shapes, diff_prediction).cpu().detach().numpy()), 'vae', np.sqrt(criterion(shapes, vae_prediction).cpu().detach().numpy()), 'gan', np.sqrt(criterion(shapes, gan_prediction).cpu().detach().numpy()))  
            print('p_thick rmse: diff', np.sqrt(criterion(p_thick, diff_predict_p_thick).cpu().detach().numpy()), 'vae', np.sqrt(criterion(p_thick, vae_predict_p_thick).cpu().detach().numpy()), 'gan', np.sqrt(criterion(p_thick, gan_predict_p_thick).cpu().detach().numpy()))
            print('shape matchShapes: diff', criterion_shape(shapes, diff_prediction)[0], 'vae', criterion_shape(shapes, vae_prediction)[0], 'gan', criterion_shape(shapes, gan_prediction)[0])
            diff_spec = forward_model_simulator(forward_model, diff_prediction, diff_predict_p_thick)
            vae_spec = forward_model_simulator(forward_model, vae_prediction, vae_predict_p_thick)
            gan_spec = forward_model_simulator(forward_model, gan_prediction, gan_predict_p_thick) 
            simis_diff = criterion_shape(shapes, diff_prediction)
            simis_vae = criterion_shape(shapes, vae_prediction)
            simis_gan = criterion_shape(shapes, gan_prediction)
            origin_p_thick = process_predicted_p_thick(p_thick)
            # for j in range(64):
            #     print(j, simis_diff[j], simis_vae[j], simis_gan[j]) 
            #     origin_shape, diff_shape, vae_shape, gan_shape = draw_image(shapes[j], diff_prediction[j], vae_prediction[j], gan_prediction[j], j)
            #     np.savez(f'val/origin_target_shape_{j}.npz', images=origin_shape.detach().numpy(), p_thick=origin_p_thick[j])
            #     np.savez(f'val/diff_target_shape_{j}.npz', images=diff_shape.detach().numpy(), p_thick=diff_predict_p_thick[j])
            #     np.savez(f'val/vae_target_shape_{j}.npz', images=vae_shape.detach().numpy(), p_thick=vae_predict_p_thick[j])
            #     np.savez(f'val/gan_target_shape_{j}.npz', images=gan_shape.detach().numpy(), p_thick=gan_predict_p_thick[j])
           