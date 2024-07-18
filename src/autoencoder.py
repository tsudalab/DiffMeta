import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(400, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 400),
            nn.Sigmoid() 
        )
        
    def forward_encoder(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            return x
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x