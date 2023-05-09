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

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 60),  # input layer
            nn.ReLU(),
            nn.Linear(60, 20),
            nn.ReLU()
        )
        self.z_mu = nn.Linear(20, 10)
        self.z_sigma = nn.Linear(20, 10)

        self.decoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 60),
            nn.ReLU(),
            nn.Linear(60, 8 * 3)  # output layer
        )

    def forward(self, x, mode="training"):
        encoded = self.encoder(x)
        z_mu = self.z_mu(encoded)
        z_sigma = self.z_sigma(encoded)
        epsilon = torch.randn_like(z_sigma)

        if mode == "training":
            z = z_mu + z_sigma * epsilon
        elif mode == "testing":
            z = self.z_mu(encoded)

        decoded = self.decoder(z)

        return decoded, z_mu, z_sigma
