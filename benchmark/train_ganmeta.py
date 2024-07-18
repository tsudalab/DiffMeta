import sys
import torch
from torch import nn
import numpy as np
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
from torch.utils.data import DataLoader
from DiffMeta.benchmark.models import cGAN, SimulatorNet_new_fc
from fastprogress import progress_bar
from pytorch_msssim import SSIM
from dataset import MetaMaterials
from torch.optim.lr_scheduler import StepLR, ExponentialLR


seed = 1024
torch.manual_seed(seed)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device', DEVICE)
# For cGAN model
def train(model, train_pbar, optimizer_G, optimizer_D, criterion, criterion_shape, configs):

    model.train()
    g_loss_epoch = 0
    d_loss_epoch = 0
    g_loss_1_epoch = 0
    g_loss_2_epoch = 0

    for img, spec, p_thick in train_pbar:

        batch_size = len(spec)
        spec, img, p_thick = spec.to(DEVICE), img.to(DEVICE), p_thick.to(DEVICE)
        
        # Ground truth
        valid = torch.ones(batch_size, 1).to(DEVICE)
        fake = torch.zeros(batch_size, 1).to(DEVICE)

        # Train the generator

        optimizer_G.zero_grad()
        z = model.sample_noise(batch_size, configs.prior).to(DEVICE)

        gen_img, gen_gap = model.Generator(spec, z)
        validity = model.Discriminator(gen_img, gen_gap, spec)

        if configs.if_struc ==0:
            g_loss_1 = criterion(validity, valid)
            g_loss_2 = 0
            g_loss = g_loss_1
            g_loss.backward()
        else:
            g_loss_1 = criterion(validity, valid)
            g_loss_2 = criterion_shape(img, gen_img)
            g_loss = g_loss_1 - configs.alpha*g_loss_2
            g_loss.backward()

        optimizer_G.step()

        # train the discriminator

        optimizer_D.zero_grad()
        # on real data

        real_pred = model.Discriminator(img, p_thick, spec)

        d_loss_real = criterion(real_pred, valid)

        # on generated data
        gen_img, gen_gap = model.Generator(spec, z)

        fake_pred = model.Discriminator(gen_img, gen_gap, spec)
        d_loss_fake = criterion(fake_pred, fake)

        d_loss = (d_loss_real + d_loss_fake)/2
        d_loss.backward()
        optimizer_D.step()

        g_loss_epoch += g_loss * batch_size
        d_loss_epoch += d_loss * batch_size
        g_loss_1_epoch += g_loss_1 * batch_size
        g_loss_2_epoch += g_loss_2 * batch_size
        train_pbar.comment = f"loss={d_loss.item():2.3f}"


    g_loss_epoch, d_loss_epoch, g_loss_1_epoch, g_loss_2_epoch = g_loss_epoch / train_pbar.total, d_loss_epoch / train_pbar.total, g_loss_1_epoch / train_pbar.total, g_loss_2_epoch / train_pbar.total

    return g_loss_epoch+d_loss_epoch


def evaluate(model, val_loader, forward_model, configs):
    model.eval()
    dataloader = val_loader

    with torch.no_grad():

        spec, img, p_thick = dataloader.dataset.target,dataloader.dataset.images, dataloader.dataset.p_thick
        
        spec, img, p_thick =  spec.to(DEVICE), img.to(DEVICE), p_thick.to(DEVICE)
        batch_size = len(spec)

        z = model.sample_noise(batch_size, configs.prior).to(DEVICE)

        gen_img, gen_gap = model.Generator(spec, z)
        spec_pred = forward_model_simulator(forward_model, gen_img, gen_gap)
        criterion_1 = nn.MSELoss()
        loss_predict = criterion_1(spec, spec_pred)

    return loss_predict


def save_checkpoint(model, optimizer_G, optimizer_D, epoch, loss_all_train, path, configs):
    # save the saved file 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'G_optimizer_state_dict': optimizer_G.state_dict(),
            'D_optimizer_state_dict': optimizer_D.state_dict(),
            'loss_all_train':loss_all_train,
            'configs':configs,
        }, path)


@torch.no_grad()
def forward_model_simulator(forward_model, shape, p_thick):  
    with torch.no_grad():
        spectrum_pred = forward_model(shape, p_thick)
    return spectrum_pred

def main(configs):

    gan_path = '../models/ganmeta.pth' 
    forword_path =  '../models/simulator.pth'

    train_dataset = MetaMaterials('train')
    val_dataset = MetaMaterials('val')
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False)
    train_pbar = progress_bar(train_loader, leave=False)
    val_pbar = progress_bar(val_loader, leave=False)

    model = cGAN(img_size=64, spec_dim=configs.spec_dim, noise_dim=configs.noise_dim, d=configs.net_depth, thickness=configs.layers, k_size=configs.k_size, k_pad=configs.k_pad).to(DEVICE)

    optimizer_G = torch.optim.Adam(model.Generator.parameters(), lr=configs.g_lr)
    optimizer_D = torch.optim.Adam(model.Discriminator.parameters(), lr=configs.d_lr)

    if configs.if_lr_de==0:
        # 0 for step case decay, 1 for exponential decay 
        scheduler_G = StepLR(optimizer_G, step_size=configs.epoch_lr_de, gamma=configs.lr_de)
        scheduler_D = StepLR(optimizer_D, step_size=configs.epoch_lr_de, gamma=configs.lr_de)

    else:
        # choose lr_de 0.9, 0.999, 1.0
        # 1 for stepcase decay, 2 for exponential decay
        scheduler_G = ExponentialLR(optimizer_G, configs.lr_de)
        scheduler_D = ExponentialLR(optimizer_D, configs.lr_de)

    criterion = torch.nn.BCELoss()
    criterion_shape = SSIM(data_range=1, size_average=True, channel=1)
    forward_model = SimulatorNet_new_fc(spec_dim=400, d=64).to(DEVICE)
    forward_model.load_state_dict(torch.load(forword_path)['model_state_dict'])
    epochs = configs.epochs
    loss_best = 1e10
    early_stop = 710
    early_temp = 0
    
    for e in range(epochs):

        loss_train = train(model, train_pbar, optimizer_G, optimizer_D, criterion, criterion_shape, configs)
        loss_val = evaluate(model, val_loader, forward_model, configs)

        if loss_best >= loss_val:
            # save the best model for smallest validation RMSE
            loss_best = loss_val
            save_checkpoint(model, optimizer_G, optimizer_D, e, loss_train, gan_path, configs)
            early_temp = 0
        else:
            early_temp +=1

        if early_temp>=early_stop:
            print('Reached early stopping, stopped straining.')
            break
        
        scheduler_D.step()
        scheduler_G.step()

        print('Epoch {}, train loss {:.4f}, val loss {:.4f}, best_val loss {:.4f}, early_stop {:d}'.format(e, loss_train, loss_val, loss_best, early_temp))
    

if __name__  == '__main__':

    parser = argparse.ArgumentParser('nn models for inverse design: cGAN')
    parser.add_argument('--model', type=str, default='cGAN')

    parser.add_argument('--img_size', type=int, default=64, help='Input size of image')
    parser.add_argument('--spec_dim', type=int, default=400, help='Input dimension of spectrum')
    parser.add_argument('--noise_dim', type=int, default=50, help='Dimension of noise variable')
    parser.add_argument('--net_depth', type=int, default=64, help='Depth of neuron layers')

    parser.add_argument('--batch_size', type=int, default=256, help='Batch size of dataset')
    parser.add_argument('--layers', nargs="+", type=int, default=[1,1,1], help='Number of layers for gap, img, spec when stack together')
    parser.add_argument('--k_size', type=int, default=5, help='size of kernel, use 3 or 5')
    parser.add_argument('--k_pad', type=int, default=2, help='size of kernel padding, use 1 for kernel=3, 2 for kernel=5')

    parser.add_argument('--prior', type=int, default=1, help='1 for (0,1) normal distribution, 0 for (0,1) uniform distribution')
    
    parser.add_argument('--if_struc', type=int, default=1, help='1 adding SSIM structure loss')
    parser.add_argument('--alpha', type=float, default=0.05, help='Factor for including structure loss')
    
    parser.add_argument('--epochs', type=int, default=5000, help='Number of iteration steps')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Decay rate for the Adams optimizer')
    parser.add_argument('--g_lr', type=float, default=1e-5, help='Learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=1e-5, help='Learning rate for discriminator')
    parser.add_argument('--if_lr_de',type=int, default=0, help='1 for step decay, 2 for exponential decay')
    parser.add_argument('--lr_de', type=float, default=0.75, help='Decrease the learning rate by this factor')
    parser.add_argument('--epoch_lr_de', type=int, default=700, help='Decrease the learning rate after epochs')
    
    
    parser.add_argument('--beta_1', type=float, default=0.5, help='Beta 1 for Adams optimization' )
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta 2 for Adams optimization' )
    args = parser.parse_args()

    main(args)