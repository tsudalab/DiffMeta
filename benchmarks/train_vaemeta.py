import torch
from torch import nn
import argparse
import os
from torch.utils.data import DataLoader
from fastprogress import progress_bar
from models import SimulatorNet_new_fc, cVAE_GSNN, cVAE_hybrid
from dataset import MetaMaterials
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from pytorch_msssim import SSIM

# Set device and seed
def setup_device(seed=1024):
    torch.manual_seed(seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    return device
# Get command-line arguments
def get_args():
    parser = argparse.ArgumentParser(description='Training script for cVAE with forward model')
    parser.add_argument('--model', type=str, default='cVAE_hybrid')
    parser.add_argument('--img_size', type=int, default=64, help='Input size of image')
    parser.add_argument('--spectrum_dim', type=int, default=400, help='Dimension of spectrum')
    parser.add_argument('--net_depth', type=int, default=16, help='Depth of convolution layers')
    parser.add_argument('--latent_dim', type=int, default=64, help='Dimension of latent variable')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size of dataset')
    parser.add_argument('--en', type=int, default=0, help='1 for data augmentation')
    parser.add_argument('--layers', type=int, nargs='+', default=[4, 1, 1], help='Number of layers for gap, img, spec')
    parser.add_argument('--k_size', type=int, default=5, help='Size of kernel')
    parser.add_argument('--k_pad', type=int, default=2, help='Size of kernel padding')
    parser.add_argument('--weight_replace', type=float, default=0.1, help='Weight of replace loss')
    parser.add_argument('--weight_vae', type=float, default=1.0, help='Weight of VAE loss')
    parser.add_argument('--weight_KLD', type=float, default=0.5, help='Weight of KLD loss')
    parser.add_argument('--weight_forward', type=float, default=1.0, help='Weight of forward model loss')
    parser.add_argument('--alpha', type=float, default=0.5, help='Factor of SSIM image loss')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of iteration steps')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Decay rate for the Adam optimizer')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--lr_de_type', type=int, default=0, help='0 for step decay, 1 for exponential decay')
    parser.add_argument('--lr_de', type=float, default=0.8, help='Learning rate decay factor')
    parser.add_argument('--epoch_lr_de', type=int, default=200, help='Epochs before learning rate decay')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta 1 for Adam optimization')
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta 2 for Adam optimization')
    return parser.parse_args()

# Initialize model, optimizers, and schedulers
def initialize_components(configs, device):
    forward_model = SimulatorNet_new_fc(spec_dim=configs.spectrum_dim, d=64).to(device)
    forward_model.load_state_dict(torch.load('../models/simulator.pth')['model_state_dict'])

    vae_model = cVAE_GSNN(
        spec_dim=configs.spectrum_dim,
        latent_dim=configs.latent_dim,
        d=configs.net_depth,
        thickness=configs.layers,
        k_size=configs.k_size,
        k_pad=configs.k_pad
    ).to(device)

    model = cVAE_hybrid(forward_model, vae_model)

    optimizer = torch.optim.Adam(
        model.vae_model.parameters(),
        lr=configs.lr,
        betas=(configs.beta_1, configs.beta_2),
        weight_decay=configs.weight_decay
    )

    scheduler = (StepLR(optimizer, step_size=configs.epoch_lr_de, gamma=configs.lr_de)
                 if configs.lr_de_type == 0
                 else ExponentialLR(optimizer, configs.lr_de))

    criterion = nn.MSELoss()
    criterion_shape = SSIM(data_range=1, size_average=True, channel=1)

    return model, optimizer, scheduler, criterion, criterion_shape

# Training function
def train(model, train_pbar, optimizer, criterion, criterion_shape, configs, device):
    model.vae_model.train()
    model.forward_model.eval()
    loss_epoch = 0

    for img, spec, p_thick in train_pbar:
        spec, img, p_thick = spec.to(device), img.to(device), p_thick.to(device)

        optimizer.zero_grad()
        img_pred, gap_pred, mu, logvar, img_hat, gap_hat, spec_pred = model(img, p_thick, spec)
        
        replace_loss = criterion(gap_hat, p_thick) - configs.alpha * criterion_shape(img, img_hat)
        vae_loss = criterion(gap_pred, p_thick) - configs.alpha * criterion_shape(img, img_pred)
        forward_loss = criterion(spec, spec_pred)
        KLD_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = (configs.weight_replace * replace_loss +
                configs.weight_vae * vae_loss +
                configs.weight_KLD * KLD_loss +
                configs.weight_forward * forward_loss)

        loss.backward()
        optimizer.step()

        loss_epoch += loss * len(spec)
        train_pbar.comment = f"Loss: {loss.item():.3f}"

    return loss_epoch / train_pbar.total

# Evaluation function
def evaluate(model, val_loader, criterion, criterion_shape, forward_model, configs, device):
    model.eval()
    with torch.no_grad():
        spec, img, gap = val_loader.dataset.target, val_loader.dataset.images, val_loader.dataset.p_thick
        gap, spec, img = gap.to(device), spec.to(device), img.to(device)

        img_pred, gap_pred, mu, logvar, img_hat, gap_hat, spec_pred = model(img, gap, spec)
        spec_pred_eval = forward_model(img_pred, gap_pred)

        replace_loss = criterion(gap_hat, gap) - configs.alpha * criterion_shape(img, img_hat)
        vae_loss = criterion(gap_pred, gap) - configs.alpha * criterion_shape(img, img_pred)
        forward_loss = criterion(spec, spec_pred)
        forward_loss_eval = criterion(spec, spec_pred_eval)
        KLD_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = (configs.weight_replace * replace_loss +
                configs.weight_vae * vae_loss +
                configs.weight_KLD * KLD_loss +
                configs.weight_forward * forward_loss)
        
        loss_eval = (configs.weight_replace * replace_loss +
                     configs.weight_vae * vae_loss +
                     configs.weight_KLD * KLD_loss +
                     configs.weight_forward * forward_loss_eval)

    return loss_eval

# Save model checkpoint
def save_checkpoint(model, optimizer, epoch, loss_train, path, configs):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_train,
        'configs': configs,
    }, path)


# Main training loop
def main(configs):
    
    device = setup_device()

    # Initialize components
    model, optimizer, scheduler, criterion, criterion_shape = initialize_components(configs, device)

    # Load datasets and dataloaders
    train_dataset = MetaMaterials('train')
    val_dataset = MetaMaterials('val')
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False)
    train_pbar = progress_bar(train_loader, leave=False)

    epochs = configs.epochs
    loss_best = float('inf')
    early_stop = 710
    early_temp = 0

    for epoch in range(epochs):
        loss_train = train(model, train_pbar, optimizer, criterion, criterion_shape, configs, device)
        loss_eval = evaluate(model, val_loader, criterion, criterion_shape, model.forward_model, configs, device)

        if loss_best >= loss_eval:
            loss_best = loss_eval
            save_checkpoint(model, optimizer, epoch, loss_eval, '../models/vaemeta.pth', configs)
            early_temp = 0
        else:
            early_temp += 1

        if early_temp >= early_stop:
            print('Reached early stopping criteria.')
            break

        scheduler.step()
        print(f'Epoch {epoch}, Train Loss: {loss_train:.6f}, Val Loss: {loss_eval:.6f}, Early Stop: {early_temp}')

if __name__ == '__main__':
    configs = get_args()
    main(configs)
