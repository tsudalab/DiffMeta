import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MetaMaterials
from diffusion import GaussianDiffusion
from unet import Unet
from fastprogress import progress_bar
import argparse

# Device setup
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
print('__Number CUDA Devices:', torch.cuda.device_count())

def get_args():
    parser = argparse.ArgumentParser(description='Training Diffusion Model')
    parser.add_argument('--spectrum_dim', type=int, default=400, help='Dimension of spectrum')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs for training')
    parser.add_argument('--early_stop', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_path', type=str, default="../models/diffmeta.pth", help='Path to save the model')
    return parser.parse_args()

def diffmeta_validation(diff_model, dataloader):
    val_loss = []
    pbar = progress_bar(dataloader, leave=False)
    for shape, spectrum, p_thick in pbar:
        spec, img, p_thick = spectrum.to(DEVICE), shape.to(DEVICE), p_thick.to(DEVICE)
        target = spec.unsqueeze(1)
        loss = diff_model(img, target, p_thick)
        loss_predict = loss.mean()
        val_loss.append(loss_predict.item())
    return np.array(val_loss).mean()

def diffmeta_train_one_epoch(diff_model, dataloader, optimizer):
    epoch_losses = []
    pbar = progress_bar(dataloader, leave=False)
    for shape, spectrum, p_thick in pbar:
        inputs = shape.to(DEVICE)
        target = spectrum.to(DEVICE)
        p_thick = p_thick.to(DEVICE)
        target = target.unsqueeze(1)
        
        optimizer.zero_grad()
        loss = diff_model(inputs, target, p_thick)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        
        pbar.comment = f"MSE={loss.item():2.3f}"
        epoch_losses.append(loss.item())
    return np.array(epoch_losses).mean()

def main(args):
    # Dataset and DataLoader setup
    train_dataset = MetaMaterials('train')
    val_dataset = MetaMaterials('val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False)
    
    # Model and optimizer setup
    diffmeta = GaussianDiffusion(
        Unet(args, dim=64, device=DEVICE, dim_mults=(1, 2, 2, 4), ),
        image_size=64, p2_loss_weight_k=0, timesteps=1000, loss_type="l2"
    ).to(DEVICE)
    unet=torch.nn.DataParallel(diffmeta.model, device_ids=[0,1,2,3])

    optimizer = optim.Adam(diffmeta.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.75, threshold=1e-4, verbose=True, min_lr=0
    )
    
    # Training loop
    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(args.epochs):
        unet.module.train()
        train_loss = diffmeta_train_one_epoch(diffmeta, train_loader, optimizer)
        unet.module.eval()
        val_loss = diffmeta_validation(diffmeta, val_loader)
        scheduler.step(train_loss)
        
        print(f"Epoch: {epoch}, train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, best_loss: {best_loss:.6f}, early_stop_counter: {early_stop_counter}")

        if val_loss < best_loss:
            early_stop_counter = 0
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffmeta.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'scheduler_state_dict': scheduler.state_dict()
            }, args.save_path)
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= args.early_stop:
            print('Reached early stopping criterion.')
            break

if __name__ == '__main__':
    args = get_args()
    main(args)
