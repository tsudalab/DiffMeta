import torch
from torch import nn
import numpy as np
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch
print('__Number CUDA Devices:', torch.cuda.device_count())
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar
from torch.utils.data import DataLoader
from DiffMeta.benchmark.models import SimulatorNet_new_fc, cVAE_GSNN, cVAE_hybrid
from DiffMeta.benchmark.dataset import MetaMaterials
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from pytorch_msssim import SSIM
seed = 1024
torch.manual_seed(seed)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device', DEVICE)

def train(model, train_pbar, optimizer, criterion, criterion_shape, configs):

    # x: structure ; y: CIE 

    model.vae_model.train()
    model.forward_model.eval()

    loss_epoch = 0

    for img, spec, p_thick in train_pbar:
        spec, img, p_thick = spec.to(DEVICE), img.to(DEVICE), p_thick.to(DEVICE)
        
        optimizer.zero_grad()
        img_pred, gap_pred, mu, logvar, img_hat, gap_hat, spec_pred =  model(img, p_thick, spec)
        replace_loss = criterion(gap_hat, p_thick) - configs.alpha * criterion_shape(img, img_hat)

        vae_loss =  criterion(gap_pred, p_thick) - configs.alpha * criterion_shape(img, img_pred)

        forward_loss = criterion(spec, spec_pred)

        KLD_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = configs.weight_replace*replace_loss + configs.weight_vae*vae_loss + configs.weight_KLD*KLD_loss + configs.weight_forward*forward_loss

        loss.backward()
        optimizer.step()

        loss_epoch += loss*len(spec)
        train_pbar.comment = f"loss={loss.item():2.3f}"

    loss_epoch = loss_epoch / train_pbar.total
    return loss_epoch


def evaluate(model, val_loader, criterion, criterion_shape, forward_model, epoch, configs):
 
    # x: structure ; y: CIE 

    model.eval()

    dataloader = val_loader

    with torch.no_grad():

        spec, img, gap = dataloader.dataset.target,dataloader.dataset.images, dataloader.dataset.p_thick
        
        gap, spec, img = gap.to(DEVICE), spec.to(DEVICE), img.to(DEVICE)

        img_pred, gap_pred, mu, logvar, img_hat, gap_hat, spec_pred  = model(img, gap, spec)

        spec_pred_eval = forward_model(img_pred, gap_pred)

        replace_loss = criterion(gap_hat, gap) - configs.alpha * criterion_shape(img, img_hat)

        vae_loss =  criterion(gap_pred, gap) - configs.alpha * criterion_shape(img, img_pred)

        forward_loss = criterion(spec, spec_pred)
        forward_loss_eval = criterion(spec, spec_pred_eval)

        KLD_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = configs.weight_replace*replace_loss + configs.weight_vae*vae_loss + configs.weight_KLD*KLD_loss + configs.weight_forward*forward_loss
        loss_eval = configs.weight_replace*replace_loss + configs.weight_vae*vae_loss + configs.weight_KLD*KLD_loss + configs.weight_forward*forward_loss_eval


    return loss_eval


def save_checkpoint(model, optimizer, epoch, loss_train, path, configs):
    # save the saved file 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':loss_train,
            'configs':configs,
        }, path)




def main(configs):
    
    vae_path = '../models/vaemeta.pth'
    configs.epoch_lr_de = 700
    configs.epochs = 5000
    train_dataset = MetaMaterials('train')
    val_dataset = MetaMaterials('val')
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False)
    train_pbar = progress_bar(train_loader, leave=False)
    forward_model = SimulatorNet_new_fc(spec_dim=400, d=64).to('cuda')
    forward_model.load_state_dict(torch.load('../models/simulator.pth')['model_state_dict'])
    vae_model = cVAE_GSNN(spec_dim=configs.spec_dim, latent_dim=configs.latent_dim, d=configs.net_depth, thickness=configs.layers, k_size=configs.k_size, k_pad=configs.k_pad).to(DEVICE)
    model = cVAE_hybrid(forward_model, vae_model)

    # set up optimizer and criterion 

    #optimizer = torch.optim.Adam(model.vae_model.parameters(), lr=configs.lr, betas=(configs.beta_1, configs.beta_2), weight_decay=configs.weight_decay)
    
    optimizer = torch.optim.Adam(model.vae_model.parameters(), lr=configs.lr)

    if configs.lr_de_type == 0:
        # set up learning rate decaying type
        scheduler = StepLR(optimizer, step_size=configs.epoch_lr_de, gamma=configs.lr_de)
    else:
        scheduler = ExponentialLR(optimizer, configs.lr_de)


    criterion = nn.MSELoss()
    criterion_shape = SSIM(data_range=1, size_average=True, channel=1)
    
    epochs = configs.epochs
    loss_best = 1e10
    early_stop = 710
    early_temp = 0
    
    
    for e in range(epochs):

        loss_train = train(model, train_pbar, optimizer, criterion, criterion_shape, configs)
        loss_eval = evaluate(model, val_loader, criterion, criterion_shape, forward_model, e, configs)

        if loss_best >= loss_eval:
            # save the best model for smallest validation RMSE
            loss_best = loss_eval
            save_checkpoint(model, optimizer, e, loss_eval, vae_path, configs)
            early_temp = 0
        else:
            early_temp += 1

        if early_temp >= early_stop:
            print('Reached early stopping, stopped straining.')
            break

        scheduler.step()

        print('Epoch {}, train loss {:.6f}, val loss {:.6f}, early_stop {:d}'.format(e, loss_train, loss_eval, early_temp))


if __name__  == '__main__':
    parser = argparse.ArgumentParser('nn models for inverse design: cVAE')
    parser.add_argument('--model', type=str, default='cVAE_hybrid')
    parser.add_argument('--img_size', type=int, default=64, help='Input size of image')
    
    parser.add_argument('--spec_dim', type=int, default=400, help='Dimension of spectrum, 58 for TEM, 29 for TE/TM')
    parser.add_argument('--spec_mode', type=int, default=0, help='0 for TM+TM, 1 for TE, 2 for TM')
    parser.add_argument('--net_depth', type=int, default=16, help='Dimension of convolution layers')

    parser.add_argument('--latent_dim', type=int, default=64, help='Dimension of latent variable')
    
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size of dataset')
    parser.add_argument('--en', type=int, default=0, help='1 for data augmentation')
    
    parser.add_argument('--layers', type=int, default=[4,1,1], help='Number of layers for gap, img, spec when stack together')
    parser.add_argument('--k_size', type=int, default=5, help='size of kernel, use 3 or 5')
    parser.add_argument('--k_pad', type=int, default=2, help='size of kernel padding, use 1 for kernel=3, 2 for kernel=5')

    parser.add_argument('--weight_replace', type=float, default=0.1, help='Weight of loss on forward model if forward model is added')
    parser.add_argument('--weight_vae', type=float, default=1.0, help='Weight of loss on forward model if forward model is added')
    parser.add_argument('--weight_KLD', type=float, default=0.5, help='Weight of loss on forward model if forward model is added')
    parser.add_argument('--weight_forward', type=float, default=1.0, help='Weight of loss on forward model if forward model is added')
    parser.add_argument('--alpha', type=float, default=0.5, help='Factor of SSIM image loss')
    


    parser.add_argument('--epochs', type=int, default=5000, help='Number of iteration steps')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Decay rate for the Adams optimizer')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for forward model')
    parser.add_argument('--lr_de_type', type=int, default=0, help='0 for step decay, 1 for exponential decay')
    parser.add_argument('--lr_de', type=float, default=0.8, help='Decrease the learning rate by this factor')
    parser.add_argument('--epoch_lr_de', type=int, default=200, help='Decrease the learning rate after epochs, only for step decay')
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta 1 for Adams optimization' )
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta 2 for Adams optimization' )
    parser.add_argument('--Num', type=int, default=0, help='Running times' )
    args = parser.parse_args()

    main(args)