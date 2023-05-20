import torch
from torch import nn
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, latent_dim, **kwargs):
        super().__init__()
        self.linear1 = nn.Linear(in_features=latent_dim, out_features=128*128*3, bias=False)
    
        # Downsampling
        self.downsampling_block1 = self.__downsampling_block(3, 128)
        self.downsampling_block2 = self.__downsampling_block(128, 256)
        # self.downsampling_block3 = self.__downsampling_block(256, 512)
        self.downsampling_block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1, bias=False),
            nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(output),
            nn.LeakyReLU()
        )

        # Upsampling
        self.upsampling_block1 = self.__upsampling_block(512, 512)
        self.upsampling_block2 = self.__upsampling_block(512, 256)
        # self.upsampling_block3 = self.__upsampling_block(256, 128)
        self.upsampling_block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.Conv2d(128, 128, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            # nn.LeakyReLU()
        )

        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def __downsampling_block(self, input, output):
        return nn.Sequential(
            nn.Conv2d(input, output, 3, stride=1, padding=1, bias=False),
            nn.Conv2d(output, output, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output),
            nn.LeakyReLU()
        )

    def __upsampling_block(self, input, output):
        return nn.Sequential(
            nn.ConvTranspose2d(input, output, 3, stride=1, padding=1, bias=False),
            nn.ConvTranspose2d(output, output, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output),
            nn.LeakyReLU()
        )

    def forward(self, features):
        out_linear = self.linear1(features)

        reshaped = torch.reshape(out_linear, (out_linear.shape[0], 3, 128, 128))
        out_downsampling = self.downsampling_block1(reshaped)
        out_downsampling = self.downsampling_block2(out_downsampling)
        out_downsampling = self.downsampling_block3(out_downsampling)

        out_upsampling = self.upsampling_block1(out_downsampling)
        out_upsampling = self.upsampling_block2(out_upsampling)
        out_upsampling = self.upsampling_block3(out_upsampling)
        
        output = self.out_layer(out_upsampling)

        return output


            

# def weight_init(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
#         nn.init.zeros_(m.bias)
# model.apply(weight_init)