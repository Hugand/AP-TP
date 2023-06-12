import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, latent_dim, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        # Upsampling
        self.upsampling_block1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.2)
        )
        self.upsampling_block2 = self.__upsampling_block(512, 256)
        self.upsampling_block3 = self.__upsampling_block(256, 128)
        self.upsampling_block4 = self.__upsampling_block(128, 64)
        self.upsampling_block5 = self.__upsampling_block(64, 32)

        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def __upsampling_block(self, input, output):
        return nn.Sequential(
            nn.ConvTranspose2d(input, output, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )

    def forward(self, features):
        out_upsampling = self.upsampling_block1(features)
        out_upsampling = self.upsampling_block2(out_upsampling)
        out_upsampling = self.upsampling_block3(out_upsampling)
        out_upsampling = self.upsampling_block4(out_upsampling)
        out_upsampling = self.upsampling_block5(out_upsampling)
        output = self.out_layer(out_upsampling)

        return output