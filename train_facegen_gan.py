import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

from modules.generator import Generator
from modules.discriminator import Discriminator
from facegen_gan import FGGAN
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_dataset(batch_size):
    with open('./faces.npy','rb') as f:
        n_imgs = batch_size * 142
        faces = np.load(f)
        faces = faces[:n_imgs]#.reshape((n_imgs, 3, 128, 128))
        print(faces.shape)
    
    dataloader = DataLoader(torch.from_numpy(faces).permute((0, 3, 1, 2)), batch_size, True, pin_memory=True)

    return dataloader

def main():
    batch_size = 64
    latent_dim = 100
    noise = np.random.normal(0,1,(1, 100))
    noise = torch.from_numpy(noise)

    dataloader = load_dataset(batch_size)

    # Define model
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    fggan = FGGAN(generator, discriminator)

    # Define optimizers
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.00005, weight_decay=1e-8)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.00005, weight_decay=1e-8)

    # Define loss functions
    generator_criterion = nn.BCELoss()
    discriminator_criterion = nn.BCELoss()

    # Compile model
    fggan.compile(generator_optimizer, discriminator_optimizer,
                generator_criterion, discriminator_criterion)

    # Train model
    fggan.fit(dataloader, epochs=30)

    torch.save(fggan.state_dict(), './fggan_final.pt')


if __name__ == '__main__':
    main()