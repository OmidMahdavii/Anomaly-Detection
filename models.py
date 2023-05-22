import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, window_size, latent_size):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            # shape: (window_size, 86)
            nn.Conv1d(window_size, window_size*2, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm1d(window_size*2),
            nn.LeakyReLU(),
            # shape: (window_size*2, 80)
            nn.Conv1d(window_size*2, window_size*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(window_size*4),
            nn.LeakyReLU(),
            # shape: (window_size*4, 40)
            nn.Conv1d(window_size*4, window_size*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(window_size*8),
            nn.LeakyReLU(),
            # shape: (window_size*8, 20)
            nn.Conv1d(window_size*8, window_size*16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(window_size*16),
            nn.LeakyReLU(),
            # shape: (window_size*16, 10)
            nn.Conv1d(window_size*16, latent_size, kernel_size=10, stride=1, padding=0)
            # shape: (latent_size, 1)
        )

    
    def forward(self, x):
        x = self.main(x)
        return x


class Decoder(nn.Module):
    def __init__(self, window_size, latent_size):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            # shape: (latent_size, 1)
            nn.ConvTranspose1d(latent_size, window_size*16, kernel_size=10, stride=1, padding=0),
            nn.BatchNorm1d(window_size*16),
            nn.ReLU(),
            # shape: (window_size*16, 10)
            nn.ConvTranspose1d(window_size*16, window_size*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(window_size*8),
            nn.ReLU(),
            # shape: (window_size*8, 20)
            nn.ConvTranspose1d(window_size*8, window_size*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(window_size*4),
            nn.ReLU(),
            # shape: (window_size*4, 40)
            nn.ConvTranspose1d(window_size*4, window_size*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(window_size*2),
            nn.ReLU(),
            # shape: (window_size*2, 80)
            nn.ConvTranspose1d(window_size*2, window_size, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
            # shape: (window_size, 86)
        )

    
    def forward(self, x):
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, window_size, latent_size):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(window_size, latent_size)
        self.classifier = nn.Sequential(nn.Linear(latent_size, 1), nn.Sigmoid())


    
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x.squeeze())
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
        super(GAN, self).__init__()
        self.autoencoder = Autoencoder(window_size, latent_size)
        self.discriminator = Discriminator(window_size, latent_size)
    
