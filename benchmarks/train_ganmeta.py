import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from fastprogress import progress_bar
from pytorch_msssim import SSIM
from models import cGAN, SimulatorNet_new_fc
from dataset import MetaMaterials
from torch.optim.lr_scheduler import StepLR, ExponentialLR

# Set device and seed
def setup_device(seed=1024):
    torch.manual_seed(seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    return device

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Training cGAN model for inverse design')
    parser.add_argument('--model', type=str, default='cGAN', help='Model name')
    parser.add_argument('--img_size', type=int, default=64, help='Input size of image')
    parser.add_argument('--spectrum_dim', type=int, default=400, help='Input dimension of spectrum')
    parser.add_argument('--noise_dim', type=int, default=50, help='Dimension of noise variable')
    parser.add_argument('--net_depth', type=int, default=64, help='Depth of neuron layers')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size of dataset')
    parser.add_argument('--layers', nargs="+", type=int, default=[1,1,1], help='Number of layers for gap, img, spec')
    parser.add_argument('--k_size', type=int, default=5, help='Size of kernel')
    parser.add_argument('--k_pad', type=int, default=2, help='Size of kernel padding')
    parser.add_argument('--prior', type=int, default=1, help='Noise distribution (1: normal, 0: uniform)')
    parser.add_argument('--if_struc', type=int, default=1, help='1 to include SSIM structure loss')
    parser.add_argument('--alpha', type=float, default=0.05, help='Factor for structure loss')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for Adam optimizer')
    parser.add_argument('--g_lr', type=float, default=1e-5, help='Learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=1e-5, help='Learning rate for discriminator')
    parser.add_argument('--if_lr_de', type=int, default=0, help='Learning rate decay type (0: step, 1: exponential)')
    parser.add_argument('--lr_de', type=float, default=0.75, help='Learning rate decay factor')
    parser.add_argument('--epoch_lr_de', type=int, default=700, help='Epochs before learning rate decay')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta 1 for Adam optimizer')
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta 2 for Adam optimizer')
    return parser.parse_args()

def train(model, train_pbar, optimizer_G, optimizer_D, criterion, criterion_shape, configs, DEVICE):
    model.train()
    g_loss_epoch, d_loss_epoch = 0, 0
    g_loss_1_epoch, g_loss_2_epoch = 0, 0

    for img, spec, p_thick in train_pbar:
        batch_size = len(spec)
        spec, img, p_thick = spec.to(DEVICE), img.to(DEVICE), p_thick.to(DEVICE)
        valid = torch.ones(batch_size, 1).to(DEVICE)
        fake = torch.zeros(batch_size, 1).to(DEVICE)

        # Train the generator
        optimizer_G.zero_grad()
        z = model.sample_noise(batch_size, configs.prior).to(DEVICE)
        gen_img, gen_gap = model.Generator(spec, z)
        validity = model.Discriminator(gen_img, gen_gap, spec)

        if configs.if_struc == 0:
            g_loss_1 = criterion(validity, valid)
            g_loss_2 = 0
            g_loss = g_loss_1
        else:
            g_loss_1 = criterion(validity, valid)
            g_loss_2 = criterion_shape(img, gen_img)
            g_loss = g_loss_1 - configs.alpha * g_loss_2

        g_loss.backward()
        optimizer_G.step()

        # Train the discriminator
        optimizer_D.zero_grad()
        real_pred = model.Discriminator(img, p_thick, spec)
        d_loss_real = criterion(real_pred, valid)

        gen_img, gen_gap = model.Generator(spec, z)
        fake_pred = model.Discriminator(gen_img, gen_gap, spec)
        d_loss_fake = criterion(fake_pred, fake)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        g_loss_epoch += g_loss.item() * batch_size
        d_loss_epoch += d_loss.item() * batch_size
        g_loss_1_epoch += g_loss_1.item() * batch_size
        g_loss_2_epoch += g_loss_2.item() * batch_size
        train_pbar.comment = f"generator_loss: {g_loss.item():.3f}, discriminator_loss: {d_loss.item():.3f}"

    g_loss_epoch /= train_pbar.total
    d_loss_epoch /= train_pbar.total
    g_loss_1_epoch /= train_pbar.total
    g_loss_2_epoch /= train_pbar.total

    return g_loss_epoch + d_loss_epoch

def evaluate(model, val_loader, forward_model, configs, DEVICE):
    model.eval()
    with torch.no_grad():
        spec, img, p_thick = val_loader.dataset.target, val_loader.dataset.images, val_loader.dataset.p_thick
        spec, img, p_thick = spec.to(DEVICE), img.to(DEVICE), p_thick.to(DEVICE)
        batch_size = len(spec)

        z = model.sample_noise(batch_size, configs.prior).to(DEVICE)
        gen_img, gen_gap = model.Generator(spec, z)
        spec_pred = forward_model_simulator(forward_model, gen_img, gen_gap)
        criterion = nn.MSELoss()
        loss_predict = criterion(spec, spec_pred)

    return loss_predict

def save_checkpoint(model, optimizer_G, optimizer_D, epoch, loss_all_train, path, configs):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'G_optimizer_state_dict': optimizer_G.state_dict(),
        'D_optimizer_state_dict': optimizer_D.state_dict(),
        'loss_all_train': loss_all_train,
        'configs': configs,
    }, path)

@torch.no_grad()
def forward_model_simulator(forward_model, shape, p_thick):
    return forward_model(shape, p_thick)

def main(configs):
    DEVICE = setup_device()
    gan_path = '../models/ganmeta.pth'
    forword_path = '../models/simulator.pth'

    train_dataset = MetaMaterials('train')
    val_dataset = MetaMaterials('val')
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False)
    train_pbar = progress_bar(train_loader, leave=False)

    model = cGAN(
        img_size=configs.img_size, spec_dim=configs.spectrum_dim, noise_dim=configs.noise_dim,
        d=configs.net_depth, thickness=configs.layers, k_size=configs.k_size, k_pad=configs.k_pad
    ).to(DEVICE)

    optimizer_G = optim.Adam(model.Generator.parameters(), lr=configs.g_lr, betas=(configs.beta_1, configs.beta_2))
    optimizer_D = optim.Adam(model.Discriminator.parameters(), lr=configs.d_lr, betas=(configs.beta_1, configs.beta_2))

    if configs.if_lr_de == 0:
        scheduler_G = StepLR(optimizer_G, step_size=configs.epoch_lr_de, gamma=configs.lr_de)
        scheduler_D = StepLR(optimizer_D, step_size=configs.epoch_lr_de, gamma=configs.lr_de)
    else:
        scheduler_G = ExponentialLR(optimizer_G, configs.lr_de)
        scheduler_D = ExponentialLR(optimizer_D, configs.lr_de)

    criterion = nn.BCELoss()
    criterion_shape = SSIM(data_range=1, size_average=True, channel=1)
    forward_model = SimulatorNet_new_fc(spec_dim=configs.spectrum_dim, d=64).to(DEVICE)
    forward_model.load_state_dict(torch.load(forword_path)['model_state_dict'])

    best_loss = float('inf')
    early_stop = 710
    early_temp = 0

    for epoch in range(configs.epochs):
        loss_train = train(model, train_pbar, optimizer_G, optimizer_D, criterion, criterion_shape, configs, DEVICE)
        loss_val = evaluate(model, val_loader, forward_model, configs, DEVICE)

        if loss_val < best_loss:
            best_loss = loss_val
            save_checkpoint(model, optimizer_G, optimizer_D, epoch, loss_train, gan_path, configs)
            early_temp = 0
        else:
            early_temp += 1

        if early_temp >= early_stop:
            print('Early stopping triggered.')
            break
        print(f'Epoch {epoch}, Train Loss: {loss_train:.6f}, Val Loss: {loss_val:.6f}, Early Stop: {early_temp}')
        scheduler_D.step()
        scheduler_G.step()
        
if __name__ == '__main__':
    configs = get_args()
    main(configs)
       
