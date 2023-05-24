
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FGGAN(nn.Module):
    def __init__(self, generator, discriminator, **kwargs):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.generator.apply(self.weights_init_normal)
        self.discriminator.apply(self.weights_init_normal)

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

        # Calculate loss
        generator_labels = torch.ones(batch_size).float().to(device)
        generator_loss = self.generator_loss_criterion(fake_output.squeeze(), generator_labels)

        # Update gradients
        generator_loss.backward()

        # Gradient clipping (exploding gradient)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1)

        self.generator_optimizer.step()

        g_loss = generator_loss.item()

        return g_loss
    
    def train_discriminator(self, X, noise, batch_size):
        generated_output = self.generator(noise).detach()
        fake_output = self.discriminator(generated_output)
        real_output = self.discriminator(X)

        # Calc losses
        discriminator_fake_loss = self.discriminator_loss_criterion(fake_output.squeeze(), torch.zeros(batch_size).float().to(device))
        discriminator_real_loss = self.discriminator_loss_criterion(real_output.squeeze(), torch.ones(batch_size).float().to(device))
        
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss

        # Update gradients
        discriminator_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1)

        self.discriminator_optimizer.step()

        d_loss = discriminator_loss.item()

        return d_loss

    def fit(self, X, epochs=10, batch_size=64, latent_dim=100):
        n_batches = len(X)
        batch_print_step = int(n_batches / 10)
        print("Training starting....")
        disp_noise = torch.from_numpy(np.random.normal(0, 1, (3, latent_dim))).float().to(device)

        for epoch in range(epochs):
            g_loss = 0
            d_loss = 0

            print(f"Epoch {epoch}/{epochs}: ", end="")
            for index, batch in enumerate(X):
                batch = batch.to(device)
                noise = torch.from_numpy(np.random.normal(0, 1, (batch_size, latent_dim))).float().to(device)

                self.generator_optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()

                g_loss += self.train_generator(noise, batch_size)
                d_loss += self.train_discriminator(batch, noise, batch_size)

                if index % batch_print_step == 0:
                    print("#", end="")

            g_loss /= n_batches
            d_loss /= n_batches

            print(f"\nGenerator loss: {g_loss}  Discriminator loss: {d_loss}")
            
            
            with torch.no_grad():
                fig, axs = plt.subplots(1, 3)
                fig.set_figwidth(12)
                fig.set_figheight(4)

                out = self.generator(disp_noise.float())
                print(out.shape)
                #for ax in axs:
                axs[0].imshow(np.array(F.to_pil_image(out[0] * 0.5 + 0.5)))
                axs[1].imshow(np.array(F.to_pil_image(out[1] * 0.5 + 0.5)))
                axs[2].imshow(np.array(F.to_pil_image(out[2] * 0.5 + 0.5)))
                plt.show()

                img = F.to_pil_image(out[0] * 0.5 + 0.5)
                img.save(f'/content/drive/MyDrive/img_outs/fggan_{epoch}.jpg')
            
            torch.save(self.state_dict(), './fggan_tmp.pt')
            
    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        # Apply initial weights to convolutional and linear layers
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0,0.02)
        return m
