import torch
from torch import nn
from torch.autograd import Variable
    
class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.conv_block1 = self.__block(3, 128)
        self.conv_block2 = self.__block(128, 128)
        self.conv_block3 = self.__block(128, 256)
        self.conv_block4 = self.__block(256, 256)
        self.conv_block5 = self.__block(256, 512)
        self.linear1 = nn.Sequential(
            nn.Linear(8192, 1),
            nn.Sigmoid()
        )

    def __block(self, input, output):
        return nn.Sequential(
            nn.Conv2d(input, output, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output),
            nn.LeakyReLU()
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