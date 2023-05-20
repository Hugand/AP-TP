import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import pickle

from modules.generator import Generator
from modules.discriminator import Discriminator
from facegen_gan import FGGAN

def main():
    noise = np.random.normal(-1,1,(1, 1, 100))
    noise = torch.from_numpy(noise)
    print(noise.shape)
 
    generator = Generator(noise.shape[2])
    discriminator = Discriminator()
    fggan = FGGAN(generator, discriminator)

    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.0001, weight_decay=1e-8)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.0001, weight_decay=1e-8)

    generator_criterion = nn.BCELoss()
    discriminator_criterion = nn.BCELoss()

    fggan.compile(generator_optimizer, discriminator_optimizer,
                  generator_criterion, discriminator_criterion)

    batch_size = 64

    with open('faces.npy','rb') as f:
        n_imgs = batch_size * 4
        faces = np.load(f)
        faces = faces[:n_imgs].reshape((n_imgs, 3, 128, 128))
        print(faces.shape)

        fggan.fit(torch.from_numpy(faces).float(), epochs=3)
        
        # [:batch_size*4].reshape((batch_size*4, 3, 128, 128)).astype(np.float) / 255.0

        # plt.imshow(faces)
        # plt.show()
    
    # fggan.fit()

    # img = fggan(noise.float())
    # print(img.shape)
    # print("OUT:", discriminator(img).detach().numpy())
    
    # plt.imshow(img.reshape((128, 128, 3)).detach().numpy())
    # plt.show()


    

if __name__ == '__main__':
    main()