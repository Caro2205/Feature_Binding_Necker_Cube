'''
Author: Tim Gerne
gernetim@gmail.com
'''


import torch
import torch.nn as nn

'''
Class to define the VAE using input cubes that consist of 8 corners each having the format: x, y, z, vis
vis is a marker that can either be:
                                    1: x, y, z are visible inputs
                                 or 0: x, y, z are set to 0
'''
'''
mode determines whether latent layer is:
                                        sampled from distribution with z_mu and z_sigma -> mode = "training" (default)
                                        is set to z_mu -> mode = "testing"
'''
'''
Model can be used with input size of 8*4 or 8*3, depending whether input corners contain a visibility marker
'''


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        hidden_sizes = [30, 15, 10]  # Hidden sizes for each LSTM layer

        self.encoder = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.encoder_layers = nn.ModuleList([nn.LSTM(hidden_sizes[i], hidden_sizes[i+1], batch_first=True) for i in range(len(hidden_sizes)-1)])

        self.decoder = nn.LSTM(hidden_sizes[-1], hidden_sizes[-2], batch_first=True)
        self.decoder_layers = nn.ModuleList([nn.LSTM(hidden_sizes[i+1], hidden_sizes[i], batch_first=True) for i in range(len(hidden_sizes)-2)])

        self.output_layer = nn.Linear(hidden_sizes[0], 8 * 3)  # Output layer

    def forward(self, x, mode="training"):
        encoded, _ = self.encoder(x)

        for layer in self.encoder_layers:
            encoded, _ = layer(encoded)

        decoded, _ = self.decoder(encoded)

        for layer in self.decoder_layers:
            decoded, _ = layer(decoded)

        decoded = self.output_layer(decoded)

        return decoded, torch.tensor(0), torch.tensor(0)