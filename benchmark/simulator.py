import numpy as np
import os
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"]='0, 1, 2'
import torch
print('__Number CUDA Devices:', torch.cuda.device_count())
import argparse

from dataset import MetaMaterials
from DiffMeta.benchmark.models import SimulatorNet_new_fc
from torch.utils.data import DataLoader
seed = 42
torch.manual_seed(seed)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(model, train_loader, optimizer, criterion):

    model.train()
    loss_epoch = 0

    for shape, spectrum, p_thick in train_loader:
        
        spectrum, shape, p_thick = spectrum.to(DEVICE), shape.to(DEVICE), p_thick.to(DEVICE)

        optimizer.zero_grad()

        spectrum_pred = model(shape, p_thick)
        loss = criterion(spectrum, spectrum_pred)
        loss.backward()
        optimizer.step()
        loss_epoch += loss*len(spectrum)
    
    loss_epoch = loss_epoch / len(train_loader.dataset)

    return loss_epoch


def evaluate(model, val_loader, criterion):

    model.eval()
    dataloader = val_loader

    with torch.no_grad():
        loss_epoch = 0
        
        for shape, spectrum, p_thick in dataloader:
            spectrum, shape, p_thick = spectrum.to(DEVICE), shape.to(DEVICE), p_thick.to(DEVICE)

            spectrum_pred = model(shape, p_thick)
            loss = criterion(spectrum, spectrum_pred)
            loss_epoch += loss*len(spectrum)
        
        loss_epoch = loss_epoch / len(dataloader.dataset)

    return loss_epoch


def save_checkpoint(model, optimizer, epoch, loss_all, path, configs):
    # save the saved file 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_all':loss_all,
            'configs':configs,
            'seed':seed,
        }, path)


def main(configs):
        
    train_dataset = MetaMaterials('train')
    val_dataset = MetaMaterials('val')
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False)

    model = SimulatorNet_new_fc(spec_dim=configs.spec_dim, d=configs.net_depth).to(DEVICE)
    
    model.weight_init(mean=0, std=0.02)
    model=nn.DataParallel(model, device_ids=[0, 1, 2])
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta_1, configs.beta_2), weight_decay=configs.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=configs.epoch_lr_de, factor=configs.lr_de, threshold=1e-4, verbose=True, min_lr=0)

    criterion = nn.MSELoss()
    
    # start training 

    path =  '../models/simulator.pth'
    epochs = configs.epochs
    loss_all = np.zeros([2, configs.epochs])
    loss_val_best = 100
    early_stop = 700
    early_temp = 0

    for e in range(epochs):

        loss_train = train(model, train_loader, optimizer, criterion)
        loss_val = evaluate(model, val_loader, criterion)
        loss_all[0,e] = loss_train
        loss_all[1,e] = loss_val

        if loss_val_best >= loss_all[1,e]:
            # save the best model for smallest validation RMSE
            loss_val_best = loss_all[1,e]
            save_checkpoint(model, optimizer, e, loss_all, path, configs)
            early_temp  = 0
        else:
            early_temp = early_temp+1

        if early_temp>=early_stop:
            print('Reached early stopping, stopped training.')
            break


        print('Epoch {}, train loss {:.6f}, val loss {:.6f}, best val loss {:.6f}, early_stop {:d}.'.format(e, loss_train, loss_val, loss_val_best, early_temp))
        scheduler.step(loss_train)

if __name__  == '__main__':

    parser = argparse.ArgumentParser('Forward model for free form spectrum prediction ')
    parser.add_argument('--model', type=str, default='forward_model_new_linear')
    parser.add_argument('--spec_dim', type=int, default=400, help='Dimension of spectrum')
    parser.add_argument('--net_depth', type=int, default=64, help='Output dimension of y')

    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size of dataset')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of iteration steps')
    # try to make sure all models use the same epochs for comparision 
    parser.add_argument('--k_size', type=int, default=5, help='size of kernel, use 3 or 5')
    parser.add_argument('--k_pad', type=int, default=2, help='size of kernel padding, use 1 for kernel=3, 2 for kernel=5')

    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Decay rate for the Adams optimizer')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for forward model')
    parser.add_argument('--lr_de_type', type=int, default=0, help='0 for step decay, 1 for exponential decay')
    parser.add_argument('--lr_de', type=float, default=0.75, help='Decrease the learning rate by this factor')
    parser.add_argument('--epoch_lr_de', type=int, default=50, help='Decrease the learning rate after epochs')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta 1 for Adams optimization' )
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta 2 for Adams optimization' )
    args = parser.parse_args()

    main(args)