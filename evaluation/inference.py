import numpy as np
import torch
import torch.nn as nn
from DiffMeta.diffmeta.dataset import MetaMaterials
from torch.utils.data import DataLoader
from DiffMeta.diffmeta.diffusion import GaussianDiffusion, Unet
from DiffMeta.diffmeta.unet import SimulatorNet_new_fc
from tabulate import tabulate
from fastprogress import progress_bar
import cv2
from DiffMeta.benchmark.models import cVAE_GSNN, cVAE_hybrid, cGAN

def inverse_transform(x):
    """Apply inverse transformation to tensor x."""
    x = (x + 1) / 2
    x = (x.clamp(-1, 1) + 1) / 2
    x[x < 0.5] = 0
    x[x >= 0.5] = 1
    return x

@torch.no_grad()
def forward_model_simulator(forward_model, shape, p_thick):
    """Generate spectrum prediction using the forward model."""
    return forward_model(shape, p_thick)

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
    """Calculate shape similarity using OpenCV's matchShapes."""
    origin_shape, predicted_shape = inverse_transform(origin_shape), inverse_transform(predicted_shape)
    origin_shape, predicted_shape = origin_shape.cpu().detach().numpy(), predicted_shape.cpu().detach().numpy()
    scores = [cv2.matchShapes(o[0] * 255, p[0] * 255, 2, 0.0) for o, p in zip(origin_shape, predicted_shape)]
    return scores

def criterion_parameters(origin_param, predicted_param):
    """Calculate MSE between original and predicted parameters."""
    criterion = nn.MSELoss()
    return np.sqrt(criterion(origin_param, predicted_param).cpu().detach().numpy())

def process_predicted_p_thick(predict_p_thick):
    """Recover and process predicted thicknesses."""
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
    batch_size = 64
    DEVICE = 'cuda'
    test_dataset = MetaMaterials('test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    pbar = progress_bar(test_loader, leave=False)
    
    # Model paths
    forward_model_path = '../models/simulator.pth'
    vae_path = '../models/vaemeta.pth'
    gan_path = '../models/ganmeta.pth'
    diff_path = '../models/diffmeta.pth'
    
    # Load models
    forward_model = SimulatorNet_new_fc(spec_dim=400, d=64).to(DEVICE)
    forward_model.load_state_dict(torch.load(forward_model_path)['model_state_dict'])
  
    print('Loading GAN model...')
    configs = torch.load(gan_path, map_location=DEVICE)['configs']
    gan_model = cGAN(img_size=64, spec_dim=configs.spec_dim, noise_dim=configs.noise_dim, d=configs.net_depth, thickness=configs.layers, k_size=configs.k_size, k_pad=configs.k_pad).to(DEVICE)
    gan_model.load_state_dict(torch.load(gan_path, map_location=DEVICE)['model_state_dict'])
    
    print('Loading VAE model...')
    configs = torch.load(vae_path, map_location=DEVICE)['configs']
    vae_gsnn = cVAE_GSNN(spec_dim=configs.spec_dim, latent_dim=configs.latent_dim, d=configs.net_depth, thickness=configs.layers, k_size=configs.k_size, k_pad=configs.k_pad).to(DEVICE)
    vae_model = cVAE_hybrid(forward_model, vae_gsnn).to(DEVICE)
    vae_model.load_state_dict(torch.load(vae_path, map_location=DEVICE)['model_state_dict'])
    
    print('Loading Diffusion model...')
    diff_states = torch.load(diff_path, map_location=DEVICE)
    diff_model = GaussianDiffusion(Unet(dim=128, device=DEVICE, dim_mults=(1, 1, 2, 2, 4)), image_size=64, p2_loss_weight_k=0, timesteps=1000, loss_type="l2", train=False).to(DEVICE)
    diff_model.load_state_dict(diff_states["model_state_dict"])
    pytorch_total_params = sum(p.numel() for p in diff_model.parameters())
    print(f'Total parameters in Diffusion model: {pytorch_total_params}')

    # Error lists
    diff_shape_error, vae_shape_error, gan_shape_error = [], [], []
    diff_param_error, vae_param_error, gan_param_error = [], [], []

    # Evaluate models
    with torch.no_grad():
        for shapes, spectrums, p_thick in pbar:
            shapes, spectrums, p_thick = shapes.to(DEVICE), spectrums.to(DEVICE), p_thick.to(DEVICE)
            origin_p_thick = process_predicted_p_thick(p_thick)
            
            # Generate predictions
            diff_prediction, diff_predict_p_thick = diff_generator(diff_model, spectrums)
            vae_prediction, vae_predict_p_thick = vae_generator(vae_model, spectrums)
            gan_prediction, gan_predict_p_thick = gan_generator(gan_model, spectrums)
            
            # Calculate errors
            diff_shape_error.extend(criterion_shape(shapes, diff_prediction))
            vae_shape_error.extend(criterion_shape(shapes, vae_prediction))
            gan_shape_error.extend(criterion_shape(shapes, gan_prediction))
            
            diff_param_error.append(criterion_parameters(origin_p_thick, diff_predict_p_thick))
            vae_param_error.append(criterion_parameters(origin_p_thick, vae_predict_p_thick))
            gan_param_error.append(criterion_parameters(origin_p_thick, gan_predict_p_thick))
    
    # Print results
    headers = ["Pattern error", "Size error"]
    rows = [
        ["DiffMeta", np.mean(diff_shape_error), np.mean(diff_param_error)],
        ["VAEMeta", np.mean(vae_shape_error), np.mean(vae_param_error)],
        ["GANMeta", np.mean(gan_shape_error), np.mean(gan_param_error)]
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid", showindex="always", numalign="center"))
