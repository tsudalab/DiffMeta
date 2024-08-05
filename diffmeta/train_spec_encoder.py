import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MetaMaterials
from autoencoder import Autoencoder
from fastprogress import progress_bar
import numpy as np

def get_device():
    """Determine the device to use for training."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def create_dataloaders(batch_size):
    """Create DataLoader instances for training and validation datasets."""
    train_dataset = MetaMaterials('train')
    val_dataset = MetaMaterials('val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_dataset

def create_model_and_optimizer(device, args):
    """Initialize the model, criterion, optimizer, and scheduler."""
    model = Autoencoder(args.spectrum_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=30, factor=0.75, threshold=1e-5, verbose=True, min_lr=0
    )
    return model, criterion, optimizer, scheduler

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    loss_tol = []
    pbar = progress_bar(train_loader, leave=False)
    
    for _, spectrum, _ in pbar:
        input_data = spectrum.to(device)
        optimizer.zero_grad()
        outputs = model(input_data)
        loss = criterion(outputs, input_data)
        loss_tol.append(loss.item())
        loss.backward()
        optimizer.step()
        pbar.comment = f"MSE={loss.item():.3f}"
    
    return np.mean(loss_tol)

def validate(model, val_dataset, criterion, device):
    """Validate the model on the validation set."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        inputs = val_dataset.target.to(device)
        outputs = model(inputs)
        val_loss = criterion(outputs, inputs).item()
    return val_loss

def save_model(model, optimizer, scheduler, epoch, loss, model_path):
    """Save the model, optimizer state, and scheduler state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'scheduler_state_dict': scheduler.state_dict()
    }, model_path)

def main(args):
    """Main training loop."""
    device = get_device()
    train_loader, val_dataset = create_dataloaders(batch_size=64)
    model, criterion, optimizer, scheduler = create_model_and_optimizer(device, args)
    
    num_epochs = 10000
    early_stop = 500
    early_temp = 0
    best_loss = 1e10
    model_name = 'spectrum_encoder'
    model_path = f"../models/{model_name}.pth"
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step(train_loss)
        
        val_loss = validate(model, val_dataset, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Early Stop Counter: {early_temp}')
        
        if val_loss < best_loss:
            early_temp = 0
            best_loss = val_loss
            save_model(model, optimizer, scheduler, epoch, train_loss, model_path)
        else:
            early_temp += 1
        
        if early_temp >= early_stop:
            print('Reached early stopping, training stopped.')
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Autoencoder on MetaMaterials dataset.")
    parser.add_argument('--spectrum_dim', type=int, required=True, help='Dimension of the spectrum')
    args = parser.parse_args()
    main(args)
