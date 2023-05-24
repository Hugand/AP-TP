import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, latent_dim, **kwargs):
        super().__init__()
        self.linear1 = nn.Linear(in_features=latent_dim, out_features=128*128*3, bias=False)
    
        # Downsampling
        self.downsampling_block1 = self.__downsampling_block(3, 128)
        self.downsampling_block2 = self.__downsampling_block(128, 256)
        self.downsampling_block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 4, stride=1, padding=2, bias=False),
            nn.Conv2d(512, 512, 4, stride=2, padding=2, bias=False),
            # nn.BatchNorm2d(output),
            nn.LeakyReLU(0.2)
        )

        # Upsampling
        self.upsampling_block1 = nn.Sequential(self.__upsampling_block(512, 512), nn.LeakyReLU(0.2))
        self.upsampling_block2 = self.__upsampling_block(512, 256)
        self.upsampling_block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=2, bias=False),
            nn.ConvTranspose2d(128, 128, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, stride=1, padding=1),
            nn.Tanh()
        )

    def __downsampling_block(self, input, output):
        return nn.Sequential(
            nn.Conv2d(input, output, 4, stride=1, padding=1, bias=False),
            nn.Conv2d(output, output, 4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.2)
        )

    def __upsampling_block(self, input, output):
        return nn.Sequential(
            nn.ConvTranspose2d(input, output, 4, stride=1, padding=1, bias=False),
            nn.ConvTranspose2d(output, output, 4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.2)
        )

    def forward(self, features):
        out_linear = self.linear1(features)

        reshaped = out_linear.view(-1, 3, 128, 128)
        
        # Downsampling group
        out_downsampling = self.downsampling_block1(reshaped)
        out_downsampling = self.downsampling_block2(out_downsampling)
        out_downsampling = self.downsampling_block3(out_downsampling)

        # Upsampling group
        out_upsampling = self.upsampling_block1(out_downsampling)
        out_upsampling = self.upsampling_block2(out_upsampling)
        out_upsampling = self.upsampling_block3(out_upsampling)
        output = self.out_layer(out_upsampling)

        return output