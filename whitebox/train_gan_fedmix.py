import os
import time
# import utils
import random
from torch.autograd import Variable
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
# from utils import *
from torch.nn import BCELoss
from torch.autograd import grad
import torchvision.utils as tvls
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vtils

from model import MNIST_Net, InverseMNISTNet, Discriminator          # here the Discriminator is the discriminator and the InverseMNISTNet is the generator

class SyntheticDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(2023)

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False) 

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)

def gradient_penalty(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = Variable(z,requires_grad=True).cuda()

    o,_ = D(z)
    g = grad(o, z, grad_outputs = torch.ones(o.size()).cuda(), create_graph = True)[0].view(z.size(0), -1)
    gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gp

if __name__ == '__main__':
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--num_epochs', type = int, default = 100)
    parser.add_argument('--lr', type = float, default = 0.0005)
    parser.add_argument('--beta1', type = float, default = 0.5)
    parser.add_argument('--beta2', type = float, default = 0.999)
    parser.add_argument('--lambda_gp', type = float, default = 10)
    parser.add_argument('--n_critic', type = int, default = 5)
    parser.add_argument('--z_dim', type = int, default = 10)
    parser.add_argument('--log_interval', type = int, default = 100)
    args = parser.parse_args()
    
    # load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    synthetic_dataset = torch.load('fedmix/mnist/0.01_data.pth')     # load synthetic data from fedmix
    train_image = synthetic_dataset['data']
    train_label = synthetic_dataset['label']
    train_dataset = SyntheticDataset(train_image, train_label)           # load synthetic data from fedmix
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # save the synthetic data as images
    vtils.save_image(train_image[:500], 'fedmix/mnist/0.01_data.png', nrow=8, padding=2)

    # build model
    G = InverseMNISTNet().cuda()  # generator
    D = Discriminator(input_size=28*28, n_class=1).cuda()  # discriminator

    # optimizers
    opt_G = optim.Adam(G.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))
    opt_D = optim.Adam(D.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))

    # loss
    # criterion = F.binary_cross_entropy()

    # train
    for epoch in range(1, args.num_epochs+1):
        for i, (x,y) in enumerate(train_dataloader):
            # train discriminator
            x = x.cuda()
            y = y.cuda()

            # train with real data
            opt_D.zero_grad()
            D_real = torch.ones(x.size(0), 1).cuda()    # should be 1 if true
            D_fake = torch.zeros(x.size(0), 1).cuda()   # should be 0 if fake
            D_real = Variable(D_real.cuda())
            D_fake = Variable(D_fake.cuda())
            D_real_result = D(x.view(-1, 28*28))
            D_real_loss = F.binary_cross_entropy(D_real_result, D_real)

            z = torch.randn(x.size(0), args.z_dim).cuda()
            x_fake = G(z)
            D_fake_result = D(x_fake.view(-1, 28*28))
            D_fake_loss = F.binary_cross_entropy(D_fake_result, D_fake)

            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            opt_D.step()

        # train generator
        # if epoch % args.n_critic == 0:
            opt_G.zero_grad()
            z = torch.randn(x.size(0), args.z_dim).cuda()
            x_fake = G(z)
            D_fake_result  = D(x_fake.view(-1, 28*28))
            g_loss = F.binary_cross_entropy(D_fake_result, D_real)
            g_loss.backward()
            opt_G.step()

            if i % args.log_interval == 0:
                print('Epoch: [%d/%d], Step: [%d/%d], D_loss: %.4f, D_real_loss: %.4f, D_fake_loss: %.4f'%(epoch, args.num_epochs, i, len(train_dataloader), D_loss.item(), D_real_loss.item(), D_fake_loss.item()))
                print('Epoch: [%d/%d], Step: [%d/%d], g_loss: %.4f'%(epoch, args.num_epochs, i, len(train_dataloader), g_loss.item()))

        # save images
        if epoch % 5 == 0:
            tvls.save_image(x_fake.data, 'fedmix/mnist/GAN_image/{}.png'.format(epoch), normalize = True)

    torch.save(G.state_dict(), 'fedmix/mnist/GAN_Generator.pkl')
    torch.save(D.state_dict(), 'fedmix/mnist/GAN_Discriminator.pkl')     