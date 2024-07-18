import numpy as np
import torch
print(torch.cuda.is_available())
print('__Number CUDA Devices:', torch.cuda.device_count())

import torch.nn as nn
import torch.optim as optim
from dataset import MetaMaterials
from torch.utils.data import DataLoader
from diffusion import *
from net_b import *
from fastprogress import progress_bar

@torch.no_grad()
def diff_validation(diff_model, dataloader):
    val_loss = []
    diff_model.model.eval()
    pbar = progress_bar(dataloader, leave=False)
    for shape, spectrum, p_thick in pbar:
        spec, img, p_thick =  spectrum.to(DEVICE), shape.to(DEVICE), p_thick.to(DEVICE)
        target = spec.unsqueeze(1)
        loss=diff_model(img, target, p_thick)
        loss_predict=loss.mean()
        val_loss.append(loss_predict.item())
    return np.array(val_loss).mean()

batch_size = 64

train_dataset = MetaMaterials('train')
val_dataset = MetaMaterials('val')
test_dataset = MetaMaterials('test')
train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
epochs=2000
DEVICE = "cuda:0"

diff=GaussianDiffusion(Unet(dim=128,device=DEVICE,dim_mults=(1,1,2,2,4)),image_size=64,p2_loss_weight_k=0,timesteps=1000,loss_type="l2").to(DEVICE)
diff=nn.DataParallel(diff, device_ids=[0,1,2,3])

optimizer=optim.Adam(diff.parameters(),lr=1e-3) #1e-3
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.75, threshold=1e-4, verbose=True, min_lr=0)

name="diffmeta"
early_stop = 2000
early_temp = 0
epoch_done=0

# optimizer.load_state_dict(loaded["optimizer_state_dict"])
# scheduler.load_state_dict(loaded["scheduler_state_dict"])

best_loss = 1e10
pbar = progress_bar(train_loader, leave=False)
for epoch in (range(epoch_done,epochs)):
    l_b=[]
    diff.model.train()
    for shape, spectrum, p_thick in pbar:
        inputs=shape.to(DEVICE)
        target=spectrum.to(DEVICE)
        p_thick=p_thick.to(DEVICE)
        target = target.unsqueeze(1)
        optimizer.zero_grad()
        loss=diff(inputs, target, p_thick)
        loss=loss.mean()
        loss.backward()
        optimizer.step()
        pbar.comment = f"MSE={loss.item():2.3f}"
        l_b.append(loss.item())

    val_loss = diff_validation(diff, val_loader)
    scheduler.step(np.array(l_b).mean())
    print("Epoch: ",epoch, "train_loss: ",np.array(l_b).mean(), "val_loss", val_loss,  "best loss: ",best_loss, "early stop: ", early_temp)
    if val_loss < best_loss:
        early_temp = 0
        best_loss = val_loss
        torch.save({'epoch': epoch,"model_state_dict":diff.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss,"scheduler_state_dict":scheduler.state_dict()}, "../models/"+name+".pth")
    else:
        early_temp += 1
    
    if early_temp >= early_stop:
        print('Reached early stopping, stopped training.')
        break
        



