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

from model import MNIST_Net, InverseMNISTNet, Discriminator

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

def inversion(args, classifier, evaluator, Generator, Disc, labels):

    gt_labels = torch.zeros(args.batch_size, 10).cuda()
    for i in range(args.batch_size):
        gt_labels[i][labels[i]] = 1       # generate one-hot labels
    print('The ground truth labels are: {}'.format(gt_labels))
    z = torch.randn(args.batch_size, args.z_dim).cuda()
    z = Variable(z, requires_grad=True).cuda()

    fake_image = Generator(z)
    # save the original image
    original_image = fake_image.clone().detach()
    tvls.save_image(original_image, 'fedmix/mnist/inversion_image/original_image.png', normalize=False, range=(-1, 1))

    optimizer = optim.Adam([z], lr=args.lr, betas=(0.5, 0.999))
    # optimizer = optim.Adam(Generator.parameters(), lr=args.lr, betas=(0.5, 0.999))      # optimizer for generator
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.num_epoch):
        optimizer.zero_grad()
        fake_image = Generator(z)
        _, fake_out = classifier(fake_image)

        D_fake = Disc(fake_image.view(-1, 28*28))     # prior result for discriminator
        prior_loss = -torch.mean(D_fake)    # prior loss
        # print('The fake_out is: {}'.format(fake_out))
        gt_loss = criterion(fake_out, gt_labels)     # gt loss
        loss = prior_loss + args.lambda_gt * gt_loss
        loss.backward()
        optimizer.step()
        
        print('epoch: {}, loss: {}, gt_loss: {}, prior_loss: {}'.format(epoch, loss.item(), gt_loss.item(), prior_loss.item()))

    # save the generated images
    fake_image = Generator(z)
    tvls.save_image(fake_image, 'fedmix/mnist/inversion_image/fake_image.png', normalize=False, range=(-1, 1))
    # save the images and label as pth file
    torch.save(fake_image.cpu().detach(), 'fedmix/mnist/inversion_image/fake_image.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--z_dim', type=int, default=10)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--lambda_gt', type=int, default=100)
    args = parser.parse_args()
 
    classifier = MNIST_Net().cuda()      # target model
    evauator = MNIST_Net().cuda()        # evaluator: evaluate the quality of the generated images, a model with high accuracy
    Generator = InverseMNISTNet().cuda()         # generator
    Disc = Discriminator(input_size=28*28, n_class=1).cuda()       # discriminator

    classifier.load_state_dict(torch.load('MNIST_Net.pth'))      # load the target model
    evauator.load_state_dict(torch.load('MNIST_Net.pth'))        # load the evaluator
    Generator.load_state_dict(torch.load('fedmix/mnist/GAN_Generator.pkl'))         # load the generator
    Disc.load_state_dict(torch.load('fedmix/mnist/GAN_Discriminator.pkl'))       # load the discriminator

    labels = torch.LongTensor(args.batch_size).random_(0, 10).cuda()   # generate random labels

    inversion(args, classifier, evauator, Generator, Disc, labels)

