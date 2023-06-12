import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.conv_block1 = self.__block(3, 64)
        self.conv_block2 = self.__block(64, 128)
        self.conv_block3 = self.__block(128, 256)
        self.conv_block4 = self.__block(256, 512)
        self.conv_block5 = self.__block(512, 64)
        self.linear1 = nn.Sequential(
            nn.Linear(1024, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def __block(self, input, output):
        return nn.Sequential(
            nn.Conv2d(input, output, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.2)
        )

    def forward(self, features):
        out_conv = self.conv_block1(features)
        out_conv = self.conv_block2(out_conv)
        out_conv = self.conv_block3(out_conv)
        out_conv = self.conv_block4(out_conv)
        out_conv = self.conv_block5(out_conv)
        
        flattened = out_conv.reshape(out_conv.size(0), -1)
        
        output = self.linear1(flattened)

        return output