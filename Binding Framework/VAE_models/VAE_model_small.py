import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size=32):
        self.input_size = input_size
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2),  # input layer
        )

        self.residual = nn.Sequential(
            nn.Linear(input_size, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 8 * 3),  # output layer
        )

    def forward(self, x, mode="training"):
        encoded = self.encoder(x)
        #z_mu = self.z_mu(encoded)
        #z_sigma = self.z_sigma(encoded)
        #epsilon = torch.randn_like(z_sigma) * 0

        #if mode == "training":
        #    z = z_mu + z_sigma * epsilon
        #elif mode == "testing":
        #    z = self.z_mu(encoded)

        residual = self.residual(x)
        encoded = encoded + residual

        decoded = self.decoder(encoded) # z # encoded

        return decoded