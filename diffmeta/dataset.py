import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torchvision.transforms import Lambda

class MetaMaterials(Dataset):
    
    def __init__(self, type='train'):
        super(MetaMaterials).__init__()
        data = np.load(f'../data/{type}.npz' )
        target = data['target']
        images = data['images'] 
        p = data['p'] 
        slab_thickness = data['slab_thickness']  
        space_thickness = data['space_thickness'] 
        bottom_thickness = data['bottom_thickness'] 
        p_normalized = self.min_max_normalize(p, 3, 8)
        slab_thickness_normalized = self.min_max_normalize(slab_thickness, 0, 0.8)
        space_thickness_normalized = self.min_max_normalize(space_thickness, 0, 1)
        bottom_thickness_normalized = self.min_max_normalize(bottom_thickness, 0, 0.2)
        p_thick = np.stack((p_normalized, slab_thickness_normalized, space_thickness_normalized, bottom_thickness_normalized), axis=-1)
        transform = Lambda(lambda t: (t/255 * 2) - 1)
        images = transform(images)
        self.images = torch.tensor(images).float()
        self.target = torch.tensor(target).float()
        self.p_thick = torch.tensor(p_thick).float()
        print(self.p_thick.size())
    
    def min_max_normalize(self, data, data_min, data_max):
        return (data - data_min) / (data_max - data_min)
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return self.images[idx, :, :, :], self.target[idx, :], self.p_thick[idx, :]

