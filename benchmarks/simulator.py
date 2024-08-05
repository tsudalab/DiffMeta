import numpy as np
import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from dataset import MetaMaterials
from models import SimulatorNet_new_fc


# Set device and seed
def setup_device(seed=1024):
    torch.manual_seed(seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    return device

# Training function
def train(model, train_loader, optimizer, criterion, DEVICE):
    model.train()
    total_loss = 0

    for shape, spectrum, p_thick in train_loader:
        spectrum, shape, p_thick = spectrum.to(DEVICE), shape.to(DEVICE), p_thick.to(DEVICE)
        optimizer.zero_grad()

        spectrum_pred = model(shape, p_thick)
        loss = criterion(spectrum, spectrum_pred)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(spectrum)
    
    average_loss = total_loss / len(train_loader.dataset)
    return average_loss

# Evaluation function
def evaluate(model, val_loader, criterion, DEVICE):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for shape, spectrum, p_thick in val_loader:
            spectrum, shape, p_thick = spectrum.to(DEVICE), shape.to(DEVICE), p_thick.to(DEVICE)
            spectrum_pred = model(shape, p_thick)
            loss = criterion(spectrum, spectrum_pred)
            total_loss += loss.item() * len(spectrum)
    
    average_loss = total_loss / len(val_loader.dataset)
    return average_loss

# Save checkpoint function
def save_checkpoint(model, optimizer, epoch, loss_all, path, configs):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_all': loss_all,
        'configs': configs,
        'seed': 42,
    }, path)

# Argument parsing
def get_args():
    parser = argparse.ArgumentParser(description='Forward model for free form spectrum prediction')
    parser.add_argument('--model', type=str, default='forward_model_new_linear')
    parser.add_argument('--spectrum_dim', type=int, default=400, help='Dimension of spectrum')
    parser.add_argument('--net_depth', type=int, default=64, help='Output dimension of y')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size of dataset')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of iteration steps')
    parser.add_argument('--k_size', type=int, default=5, help='Size of kernel, use 3 or 5')
    parser.add_argument('--k_pad', type=int, default=2, help='Size of kernel padding, use 1 for kernel=3, 2 for kernel=5')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Decay rate for the Adam optimizer')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for forward model')
    parser.add_argument('--lr_de_type', type=int, default=0, help='0 for step decay, 1 for exponential decay')
    parser.add_argument('--lr_de', type=float, default=0.75, help='Decrease the learning rate by this factor')
    parser.add_argument('--epoch_lr_de', type=int, default=50, help='Decrease the learning rate after epochs')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta 1 for Adam optimization')
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta 2 for Adam optimization')
    return parser.parse_args()

# Main function
def main(configs):
    # Prepare data loaders
    DEVICE = setup_device()
    train_dataset = MetaMaterials('train')
    val_dataset = MetaMaterials('val')
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False)

    # Initialize model, optimizer, and scheduler
    model = SimulatorNet_new_fc(spec_dim=configs.spectrum_dim, d=configs.net_depth).to(DEVICE)
    model.weight_init(mean=0, std=0.02)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta_1, configs.beta_2), weight_decay=configs.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=configs.epoch_lr_de, factor=configs.lr_de, threshold=1e-4, verbose=True, min_lr=0)
    criterion = nn.MSELoss()

    # Training loop
    path = '../models/simulator.pth'
    loss_all = np.zeros([2, configs.epochs])
    loss_val_best = float('inf')
    early_stop = 700
    early_temp = 0

    for epoch in range(configs.epochs):
        loss_train = train(model, train_loader, optimizer, criterion, DEVICE)
        loss_val = evaluate(model, val_loader, criterion, DEVICE)
        loss_all[0, epoch] = loss_train
        loss_all[1, epoch] = loss_val

        if loss_val < loss_val_best:
            loss_val_best = loss_val
            save_checkpoint(model, optimizer, epoch, loss_all, path, configs)
            early_temp = 0
        else:
            early_temp += 1

        if early_temp >= early_stop:
            print('Reached early stopping, stopped training.')
            break

        print(f'Epoch {epoch}, train loss {loss_train:.6f}, val loss {loss_val:.6f}, best val loss {loss_val_best:.6f}, early stop {early_temp}.')
        scheduler.step(loss_train)

if __name__ == '__main__':
    args = get_args()
    main(args)
