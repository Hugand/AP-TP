import numpy as np # linear algebra
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.transforms.functional as F
from torch.optim.lr_scheduler import ExponentialLR

class cFGGAN(nn.Module):
    def __init__(self, generator, discriminator, n_classes=2, n_embbed=32, **kwargs):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.n_classes = n_classes
        self.n_embbed = n_embbed

        self.generator_conditional_head = nn.Embedding(self.n_classes, self.n_embbed).to(device)
        self.linear_gen = nn.Sequential(
            nn.ConvTranspose2d(self.generator.latent_dim+n_embbed, 512, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        ).to(device)
        self.discriminator_conditional_head = nn.Embedding(self.n_classes, self.n_embbed).to(device)
        self.linear_disc = nn.Sequential(
            nn.Linear(1024+self.n_embbed, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 1),
            nn.Sigmoid()
        ).to(device)

        self.generator_conditional_head.apply(self.weights_init_normal)
        self.linear_gen.apply(self.weights_init_normal)
        self.discriminator_conditional_head.apply(self.weights_init_normal)
        self.linear_disc.apply(self.weights_init_normal)
        #.generator.apply(self.weights_init_normal)
        #self.discriminator.apply(self.weights_init_normal)
        
        self.d_losses = []
        self.g_losses = []
        self.schedule = False

    def __conditional_head(self):
        return nn.Sequential(
            nn.Embedding(self.n_classes, 25),
            nn.Linear(25, 128*128*1)
        )

    def get_generator_parameters(self):
        return list(self.generator_conditional_head.parameters()) + \
               list(self.linear_gen.parameters()) + \
               list(self.generator.upsampling_block2.parameters()) + \
               list(self.generator.upsampling_block3.parameters()) + \
               list(self.generator.upsampling_block4.parameters()) + \
               list(self.generator.upsampling_block5.parameters()) + \
               list(self.generator.out_layer.parameters())
        

    def get_discriminator_parameters(self):
        return list(self.discriminator_conditional_head.parameters()) + \
               list(self.discriminator.conv_block1.parameters()) + \
               list(self.discriminator.conv_block2.parameters()) + \
               list(self.discriminator.conv_block3.parameters()) + \
               list(self.discriminator.conv_block4.parameters()) + \
               list(self.discriminator.conv_block5.parameters()) + \
               list(self.linear_disc.parameters())

    def forward(self, features, class_):
        # Conditional input
        out = self.generator_conditional_head(class_).view(len(class_), self.n_embbed, 1, 1)
        # GAN input
        out_linear = self.linear_gen(torch.cat((features, out), 1))
        # Upsampling group
        out_upsampling = self.generator.upsampling_block2(out_linear) 
        out_upsampling = self.generator.upsampling_block3(out_upsampling)
        out_upsampling = self.generator.upsampling_block4(out_upsampling)
        out_upsampling = self.generator.upsampling_block5(out_upsampling)
        output = self.generator.out_layer(out_upsampling)

        return output

    def forward_discriminator(self, features, class_):
        # Conditional input
        out = self.discriminator_conditional_head(class_)
        
        out_conv = self.discriminator.conv_block1(features)
        out_conv = self.discriminator.conv_block2(out_conv)
        out_conv = self.discriminator.conv_block3(out_conv)
        out_conv = self.discriminator.conv_block4(out_conv)
        out_conv = self.discriminator.conv_block5(out_conv)
        
        flattened = out_conv.reshape(out_conv.size(0), -1)
        
        output = self.linear_disc(torch.cat((flattened, out), 1))
        
        return output # self.discriminator(conditioned_out)
    
    def compile(self,
            generator_optimizer,
            discriminator_optimizer,
            generator_loss_criterion,
            discriminator_loss_criterion,
            schedule=False
        ):
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss_criterion = generator_loss_criterion
        self.discriminator_loss_criterion = discriminator_loss_criterion
        self.schedule = schedule
        if schedule:
            self.generator_optimizer_scheduler = ExponentialLR(self.generator_optimizer, gamma=0.1, last_epoch=-1, verbose=False)
            self.discriminator_optimizer_scheduler = ExponentialLR(self.discriminator_optimizer, gamma=0.1, last_epoch=-1, verbose=False)


    def train_generator(self, noise, batch_size):
        self.generator_optimizer.zero_grad()
        
        #random_labels = torch.zeros(batch_size, self.n_classes, 1, 1)
        random_labels = torch.randint(0, 2, (batch_size,)).to(device)
        generated_output = self(noise, random_labels.int().to(device))
        fake_output = self.forward_discriminator(generated_output, random_labels.int())

        # Calculate loss
        #generator_labels = torch.ones(batch_size).float().to(device)
        generator_labels = torch.from_numpy(np.array([0.9] * batch_size)).float().to(device)
        generator_loss = self.generator_loss_criterion(fake_output.squeeze(), generator_labels)

        # Update gradients
        generator_loss.backward()

        # Gradient clipping (exploding gradient)
        torch.nn.utils.clip_grad_norm_(self.get_generator_parameters(), 1)

        self.generator_optimizer.step()
        if self.schedule:
            self.generator_optimizer_scheduler.step()

        g_loss = generator_loss.item()

        return g_loss
    
    def train_discriminator(self, X, labels, noise, batch_size):
        self.discriminator_optimizer.zero_grad()
        
        #random_labels = torch.zeros(batch_size, self.n_classes, 1, 1)
        random_labels = torch.randint(0, 2, (batch_size,)).to(device)

        generated_output = self(noise, random_labels.int()).detach()
        fake_output = self.forward_discriminator(generated_output, random_labels.int())
        
        flip_indices = torch.randperm(len(labels))[:20]
        labels[flip_indices] = labels[flip_indices] * (-1) + 1
        
        real_output = self.forward_discriminator(X, labels.int())

        # Calc losses
        fake_labels = torch.from_numpy(np.array([0.1] * batch_size)).float().to(device)
        real_labels = torch.from_numpy(np.array([0.9] * batch_size)).float().to(device)
        real_labels[flip_indices] = 0.1
        
        #discriminator_fake_loss = self.discriminator_loss_criterion(fake_output.squeeze(), torch.zeros(batch_size).float().to(device))
        #discriminator_real_loss = self.discriminator_loss_criterion(real_output.squeeze(), torch.ones(batch_size).float().to(device))
        discriminator_fake_loss = self.discriminator_loss_criterion(fake_output.squeeze(), fake_labels)
        discriminator_real_loss = self.discriminator_loss_criterion(real_output.squeeze(), real_labels)
        
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss

        # Update gradients
        discriminator_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.get_discriminator_parameters(), 1)

        self.discriminator_optimizer.step()
        if self.schedule:
            self.discriminator_optimizer_scheduler.step()

        d_loss = discriminator_loss.item()

        return d_loss

    def fit(self, X, epochs=10, batch_size=64, latent_dim=100, n_disc=1, n_gen=1):
        n_batches = len(X)
        batch_print_step = int(n_batches / 10)
        print("Training starting....", batch_size)
        #disp_noise = torch.from_numpy(np.random.normal(0, 1, (4, latent_dim))).float().to(device)
        disp_noise = torch.randn(4, latent_dim, 1, 1, device=device).float()
        disp_labels = torch.from_numpy(np.array([1, 1, 0, 0])).int().to(device)

        for epoch in range(epochs):
            g_loss = 0
            d_loss = 0

            print(f"Epoch {epoch}/{epochs}: ", end="")
            for index, (batch, labels) in enumerate(X):
                batch = batch.to(device)
                #noise = torch.from_numpy(np.random.normal(0, 1, (batch_size, latent_dim))).float().to(device)
                noise = torch.randn(batch_size, latent_dim, 1, 1, device=device).float()

                #self.train_discriminator(batch, labels.to(device), noise, batch_size)
                for i in range(n_disc):
                    d_loss_tmp = self.train_discriminator(batch, labels.to(device), noise, batch_size)
                d_loss += d_loss_tmp
                
                for i in range(n_gen):
                    g_loss_tmp = self.train_generator(noise.to(device), batch_size)
                g_loss += g_loss_tmp
                    
                if index % batch_print_step == 0:
                    print("#", end="")

            g_loss /= n_batches
            d_loss /= n_batches
            
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss)

            print(f"\nGenerator loss: {g_loss}  Discriminator loss: {d_loss}")
            
            
            with torch.no_grad():
                fig, axs = plt.subplots(1, 4)
                fig.set_figwidth(16)
                fig.set_figheight(4)
                out = self(disp_noise.float(), disp_labels.int())
                print(out.shape)
                #for ax in axs:
                axs[0].imshow(np.array(F.to_pil_image(out[0] * 0.5 + 0.5)))
                axs[1].imshow(np.array(F.to_pil_image(out[1] * 0.5 + 0.5)))
                axs[2].imshow(np.array(F.to_pil_image(out[2] * 0.5 + 0.5)))
                axs[3].imshow(np.array(F.to_pil_image(out[3] * 0.5 + 0.5)))
                plt.show()

                img = F.to_pil_image(out[0] * 0.5 + 0.5)
                #img.save(f'/content/drive/MyDrive/img_outs/fggan_{epoch}.jpg')
            
            torch.save(self.state_dict(), './cfggan_tmp.pt')
            if (epoch+1) % 5 == 0:
                torch.save(self.state_dict(), f'./cfggan_epoch_{epoch+1}.pt')
                with open(f'./cfggan_losses_{epoch+1}.npy', 'wb') as f:
                    np.save(f, np.array([self.g_losses, self.d_losses]))
            
    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        # Apply initial weights to convolutional and linear layers
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0,0.02)
        if isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=0.02)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        return m
