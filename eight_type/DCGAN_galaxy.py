import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.datasets import DatasetFolder
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm as tqdm
import torch.nn.functional as F
from torchsampler import ImbalancedDatasetSampler
import torchvision.utils as vutils

class Generator(nn.Module):
    def __init__(self, input_size, g_hidden):
        super(Generator, self, ).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_size, g_hidden * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_hidden * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(g_hidden * 8, g_hidden * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(g_hidden * 4, g_hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(g_hidden * 2, g_hidden, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_hidden),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(g_hidden, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, inputs):
        return self.main(inputs)
    
    
    
class Discriminator(nn.Module):
    def __init__(self, input_size, d_hidden):
        super(Discriminator, self).__init__(input_size, d_hidden)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, d_hidden, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(d_hidden, d_hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(d_hidden * 2, d_hidden * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_hidden * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(d_hidden * 4, d_hidden * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_hidden * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(d_hidden * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.main(inputs)

def train():
    # hyperparameter
    batch_size = 32
    lr_generator = 0.0001
    lr_discriminator = 0.0001
    in_img = 196608
    epochs = 100
    random.seed(999)
    torch.manual_seed(999)

    g_hidden = 64
    d_hidden = 64
    input_size = 100
    image_size = 64

    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print("CUDA is not available")
    else:
        print("CUDA is available")
    device = 'cuda' if train_on_gpu else 'cpu'

    train_image_path = "/home/chisc/workspace/wuzhenrong/galaxy/three_final/train/gans/"
    train_trans = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    train_data = ImageFolder(train_image_path, transform = train_trans)
    train_loader = DataLoader(train_data, pin_memory = True, batch_size = batch_size, shuffle=True)

    def imshow(imgs):
        imgs = torchvision.utils.make_grid(imgs, padding=2, normalize=True)
        npimgs = imgs.numpy()
        plt.figure(figsize=(8,8))
        plt.imshow(np.transpose(npimgs, (1,2,0)))
        plt.xticks([])
        plt.yticks([])
        plt.show()

    img, label = next(iter(train_loader))
    # imshow(img)
    # plt.xticks([])
    # plt.yticks([])
    imshow(img)
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    ## all model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02.
    def weight_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1: ## convolutional layer
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1: ## batchnormal layer
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    G = Generator(input_size, g_hidden).to(device)
    D = Discriminator(input_size, d_hidden).to(device)
    G.apply(weight_init)
    D.apply(weight_init)
    print(G)
    print(D)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    d_opt = optim.Adam(D.parameters(), lr=lr_generator, betas=(0.5, 0.999))
    g_opt = optim.Adam(G.parameters(), lr=lr_discriminator, betas=(0.5, 0.999))


    img_list = []
    G_losses = []
    D_losses = []

    ## Training
    for epoch in range(epochs):
        g_total_loss = 0.0
        d_total_loss = 0.0
        for idx, data in tqdm(enumerate(train_loader)):
            # update d network
            D.zero_grad()
            real = data[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, device=device)
            output = D(real).view(-1)
    #         print(output.shape)
    #         print('label:', label.shape)
            d_real_loss = criterion(output, label)
            d_real_loss.backward()
            D_x = output.mean().item()
            
            noise = torch.randn(b_size, 100, 1, 1, device=device)
            fake = G(noise)
            label.fill_(fake_label)
            output = D(fake.detach()).view(-1)
            
            d_fake_loss = criterion(output, label)
            d_fake_loss.backward()
            
            D_G_z1 = output.mean().item()
            d_loss = d_real_loss + d_fake_loss
            d_opt.step()
            
            # update g network
            G.zero_grad()
            label.fill_(real_label)
            output = D(fake).view(-1)
            g_loss = criterion(output, label)
            g_loss.backward()
            D_G_z2 = output.mean().item()
            g_opt.step()
            
            # Output training stats
            if idx % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, epochs, idx, len(train_loader), d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))
                samples = G(noise).detach()
                samples = samples.view(samples.size(0), 3, 64, 64).cpu()
                imshow(samples)
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

    torch.save(G, 'S_type_generator.pkl')
    torch.save(D, 'S_type_discriminator.pkl')

    noise = torch.randn(100, 100, 1, 1, device=device)
    samples = G(noise).detach()
    samples = samples.view(samples.size(0), 3, 64, 64).cpu()
    imshow(samples)

if __name__ == '__main__':
    train()