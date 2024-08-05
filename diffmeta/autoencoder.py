import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_size: int):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
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
            nn.Linear(256, input_size),
            nn.Sigmoid()  
        )
        
        # Optional: Initialize weights (e.g., Xavier initialization)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Passes input through the encoder part of the autoencoder."""
        x = self.encoder(x)
        return x
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes input through the full autoencoder (encoder + decoder)."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
