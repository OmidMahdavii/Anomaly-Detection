import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, window_size, latent_size):
        super(Encoder, self).__init__()
        self.window_size = window_size
        self.latent_size = latent_size
    
    def forward(self, x):
        # input shape: (window_size, 86), output shape: (latent_size, )
        pass


class Decoder(nn.Module):
    def __init__(self, window_size, latent_size):
        super(Decoder, self).__init__()
        self.window_size = window_size
        self.latent_size = latent_size
    
    def forward(self, x):
        # input shape: (latent_size, ), output shape: (window_size, 86)
        pass


class Discriminator(nn.Module):
    def __init__(self, window_size, latent_size):
        super(Discriminator, self).__init__()
        self.window_size = window_size
        self.latent_size = latent_size
    
    def forward(self, x):
        # input shape: (window_size, 86), output shape: (2, )
        pass


class Autoencoder(nn.Module):
    def __init__(self, window_size, latent_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(window_size, latent_size)
        self.decoder = Decoder(window_size, latent_size)
    
    def forward(self, x):
        pass


class GAN(nn.Module):
    def __init__(self, window_size, latent_size):
        super(Autoencoder, self).__init__()
        self.autoencoder = Autoencoder(window_size, latent_size)
        self.discriminator = Discriminator(window_size, latent_size)
    
    def forward(self, x):
        # return both autoencoder and discriminator outputs
        pass