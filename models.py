import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, window_size, latent_size):
        super(Encoder, self).__init__()
        # self.window_size = window_size
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(86,32,3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv2d(32,16,3),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv2d(16,8,3),
            nn.BatchNorm1d(8),
            nn.ReLU()
            
        )

    
    
    def forward(self, x):
        # input shape: (window_size, 86), output shape: (latent_size, )
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self, window_size, latent_size):
        super(Decoder, self).__init__()
        self.window_size = window_size
        self.latent_size = latent_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8,16,3),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 32,3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 42, 3),
            nn.BatchNorm1d(42),
            nn.ReLU(),

            nn.ConvTranspose2d(42,86,3),
            nn.Tanh(),
        )
    
    def forward(self, x):
        # input shape: (latent_size, ), output shape: (window_size, 86)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, window_size, latent_size):
        super(Discriminator, self).__init__()
        self.window_size = window_size
        self.latent_size = latent_size
        self.discriminator = nn.Sequential(
            nn.Conv2d(86,32,3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32,16,3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Conv2d(16,8,3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),

            nn.Conv2d(8,2,3),
            nn.BatchNorm1d(2),
            nn.LeakyReLU()
            
        )

    
    def forward(self, x):
        # input shape: (window_size, 86), output shape: (2, )
        x = self.discriminator(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, window_size, latent_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(window_size, latent_size)
        self.decoder = Decoder(window_size, latent_size)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class GAN(nn.Module):
    def __init__(self, window_size, latent_size):
        super(Autoencoder, self).__init__()
        self.autoencoder = Autoencoder(window_size, latent_size)
        self.discriminator = Discriminator(window_size, latent_size)
    
    def forward(self, x):
        # return both autoencoder and discriminator outputs
        x = self.autoencoder(x)
        y = self.discriminator(x)
        return x,y