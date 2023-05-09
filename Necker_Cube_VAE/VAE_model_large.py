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
            nn.Linear(input_size, 1000),  # input layer
            nn.ReLU(),
            nn.Linear(1000, 800),
            nn.ReLU(),
            nn.Linear(800, 600),
            nn.ReLU(),
            nn.Linear(600, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU()
        )
        self.z_mu = nn.Linear(200, 100)
        self.z_sigma = nn.Linear(200, 100)

        self.decoder = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 600),
            nn.ReLU(),
            nn.Linear(600, 800),
            nn.ReLU(),
            nn.Linear(800, 1000),
            nn.ReLU(),
            nn.Linear(1000, 8 * 3)  # output layer
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
