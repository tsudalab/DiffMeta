import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MetaMaterials
from autoencoder import Autoencoder
from fastprogress import progress_bar
import numpy as np
print(torch.cuda.is_available())
print('__Number CUDA Devices:', torch.cuda.device_count())

train_dataset = MetaMaterials('train')
val_dataset = MetaMaterials('val')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

DEVICE = "cuda:1"
model = Autoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.75, threshold=1e-5, verbose=True, min_lr=0)
num_epochs = 10000
early_stop = 500
early_temp = 0
best_loss = 1e10
name = 'spectrum_encoder'
pbar = progress_bar(train_loader, leave=False)

for epoch in range(num_epochs):
    model.train()
    loss_tol = []
    for _, spectrum, _ in pbar:
        input=spectrum.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, input)
        loss_tol.append(loss.item())
        loss.backward()
        optimizer.step()
        pbar.comment = f"MSE={loss.item():2.3f}"
    scheduler.step(np.array(loss_tol).mean())

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        inputs = val_dataset.target.to(DEVICE)
        outputs = model(inputs)
        val_loss = criterion(outputs, inputs).item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {np.array(loss_tol).mean()}, Validation Loss: {val_loss}, early stop: {early_temp}')
    if val_loss < best_loss:
        early_temp = 0
        best_loss = val_loss
        torch.save({'epoch': epoch,"model_state_dict":model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss,"scheduler_state_dict":scheduler.state_dict()}, "../models/"+name+".pth")
    else:
        early_temp += 1
    
    if early_temp >= early_stop:
        print('Reached early stopping, stopped training.')
        break
