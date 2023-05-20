import torch
from torch import nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FGGAN(nn.Module):
    def __init__(self, generator, discriminator, **kwargs):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.generator.apply(self.__weights_init)

    def forward(self, features):
        return self.generator(features)
    
    def compile(self,
            generator_optimizer,
            discriminator_optimizer,
            generator_loss_criterion,
            discriminator_loss_criterion
        ):
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss_criterion = generator_loss_criterion
        self.discriminator_loss_criterion = discriminator_loss_criterion

    def train_generator(self, noise, batch_size):
        generated_output = self.generator(noise)
        fake_output = self.discriminator(generated_output)

        # Calc losses
        generator_labels = torch.from_numpy(np.array([[1]] * batch_size, dtype=np.float32)).to(device)
        generator_loss = self.generator_loss_criterion(fake_output.float(), generator_labels)

        # Update gradients
        generator_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm(self.generator.parameters(), 1)

        self.generator_optimizer.step()

        g_loss = generator_loss.item()

        return g_loss
    
    def train_discriminator(self, X, noise, batch_size):
        generated_output = self.generator(noise)
        fake_output = self.discriminator(generated_output)
        real_output = self.discriminator(X)

        # Calc losses
        discriminator_labels = torch.from_numpy(np.array(([[0]] * batch_size) + ([[1]] * batch_size), dtype=np.float))
        discriminator_loss = self.discriminator_loss_criterion(fake_output + real_output, discriminator_labels)

        # Update gradients
        discriminator_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm(self.discriminator.parameters(), 1)

        self.discriminator_optimizer.step()

        d_loss = discriminator_loss.item()

        return d_loss

    def fit(self, X, epochs=10, batch_size=64, latent_dim=100):
        n_batches = int(len(X) / batch_size)
        batch_print_step = n_batches / 10
        print("Training starting....")

        for epoch in range(epochs):
            g_loss = 0
            d_loss = 0

            print(f"Epoch {epoch}/{epochs}: ", end="")
            for batch in range(n_batches):
                noise = torch.from_numpy(np.random.normal(0, 1, (batch_size, latent_dim))).float().to(device)

                self.generator_optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()

                g_loss += self.train_generator(noise, batch_size)
                d_loss += self.train_discriminator(X[batch], noise, batch_size)

                if batch % batch_print_step == 0:
                    print("#", end="")
                else:
                    print(".", end="")

            g_loss /= n_batches
            d_loss /= n_batches

            print(f"\nGenerator loss: {g_loss}  Discriminator loss: {d_loss}")

    def __weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)


        
