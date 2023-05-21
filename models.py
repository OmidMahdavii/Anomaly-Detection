import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, window_size, latent_size):
        super(Encoder, self).__init__()
        # self.window_size = window_size
        self.latent_size = latent_size

    
    def forward(self, x):
        # input shape: (window_size, 86), output shape: (latent_size, )
        x = nn.Sequential(
            nn.Conv1d(86,52,3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(52,32,3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32,16,3),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16,self.latent_size,3),
            nn.BatchNorm1d(self.latent_size),
            nn.ReLU()
            
        )
        return x


class Decoder(nn.Module):
    def __init__(self, window_size, latent_size):
        super(Decoder, self).__init__()
        self.window_size = window_size
        self.latent_size = latent_size

    
    def forward(self, x):
        # input shape: (latent_size, ), output shape: (window_size, 86)
        x  = nn.Sequential(
            nn.ConvTranspose1d(self.latent_size,16,3),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.ConvTranspose1d(16, 32,3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(32, 42, 3),
            nn.BatchNorm1d(42),
            nn.ReLU(),

            nn.ConvTranspose1d(42,86,3),
            nn.Tanh(),
        )
        return x


class Discriminator(nn.Module):
    def __init__(self, window_size, latent_size):
        super(Discriminator, self).__init__()
        self.window_size = window_size
        self.latent_size = latent_size


    
    def forward(self, x):
        # input shape: (window_size, 86), output shape: (2, )
        x = nn.Sequential(
            nn.Conv1d(86,32,3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32,16,3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Conv1d(16,8,3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),

            nn.Conv1d(8,2,3),
            nn.BatchNorm1d(2),
            nn.LeakyReLU()
            )
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
<<<<<<< HEAD
    
=======
    
    def forward(self, x):
        # return both autoencoder and discriminator outputs
        y = self.autoencoder(x)
        l0 = self.discriminator(y)
        l1 = self.discriminator(x)
        return y, l0, l1
>>>>>>> 3420b54d24ff50d8c844a4318f83dbef3c36fd54
