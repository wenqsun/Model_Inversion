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

from model import MNIST_Net, InverseMNISTNet, Discriminator, FedAvgCNN, InverseCIFAR10Net, resnet18

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
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
same_seeds(2023)

def inversion(args, classifier, evaluator, Generator, Disc, labels):

    gt_labels = torch.zeros(args.batch_size, 100).cuda()
    for i in range(args.batch_size):
        gt_labels[i][labels[i]] = 1       # generate one-hot labels
    print('The ground truth labels are: {}'.format(gt_labels))
    z = torch.randn(args.batch_size, args.z_dim).cuda()
    z = Variable(z, requires_grad=True).cuda()
    z = torch.concat((z, gt_labels), dim=1)     # concatenate the one-hot labels to z

    fake_image = Generator(z)
    # save the original image
    original_image = fake_image.clone().detach()
    tvls.save_image(original_image, 'result/'+args.FL_algorithm+'/inversion_image/original_image.png', normalize=False, range=(-1, 1))

    # optimizer = optim.Adam([z], lr=args.lr, betas=(0.5, 0.999))
    optimizer = optim.Adam(Generator.parameters(), lr=args.lr, betas=(0.5, 0.999))      # optimizer for generator
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.num_epoch):
        optimizer.zero_grad()
        fake_image = Generator(z)
        _, fake_out = classifier(fake_image)

        D_fake = Disc(fake_image.view(-1, 3*32*32))     # prior result for discriminator
        prior_loss = -torch.mean(D_fake)    # prior loss
        # print('The fake_out is: {}'.format(fake_out))
        gt_loss = criterion(fake_out, gt_labels)     # gt loss
        TV_loss = TV_prior(fake_image)      # total variation prior
        L2_loss = torch.norm(fake_image, 2)     # L2 loss
        loss = 0 * prior_loss + args.lambda_gt * gt_loss + args.lambda_TV * TV_loss + args.lambda_L2 * L2_loss
        loss.backward()
        optimizer.step()
        
        print('epoch: {}, loss: {}, gt_loss: {}, prior_loss: {}, TV_loss: {}, L2_loss: {}'.format(epoch, loss.item(), gt_loss.item(), prior_loss.item(), TV_loss.item(), L2_loss.item()))

    # save the generated images
    fake_image = Generator(z)
    tvls.save_image(fake_image, 'result/'+args.FL_algorithm+'/inversion_image/fake_image.png', normalize=False, range=(-1, 1))
    # save the images and label as pth file
    torch.save(fake_image, 'result/'+args.FL_algorithm+'/inversion_image/fake_image.pth')

    # test the accuracy of the generated images
    out, _ = classifier(fake_image)
    out = torch.exp(out)
    print('The output of the generated images is: {}'.format(out))

def TV_prior(image):
    # compute the total variation prior
    # image: [batch_size, 3, 32, 32]
    diff1 = image[:,:,:,:-1] - image[:,:,:,1:]
    diff2 = image[:,:,:-1,:] - image[:,:,1:,:]
    diff3 = image[:,:,1:,:-1] - image[:,:,:-1,1:]
    diff4 = image[:,:,:-1,:-1] - image[:,:,1:,1:]

    return torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--lambda_gt', type=int, default=1e7)
    parser.add_argument('--lambda_TV', type=int, default=1)
    parser.add_argument('--lambda_L2', type=int, default=1)
    parser.add_argument('--FL_algorithm', type=str, default='FedAvg')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier = resnet18().cuda()      # target model
    evauator = resnet18().cuda()      # evaluator: evaluate the quality of the generated images, a model with high accuracy
    Generator = InverseCIFAR10Net(z_dim = args.z_dim, num_class=100).cuda()         # generator
    Disc = Discriminator(input_size=3*32*32, n_class=1).cuda()       # discriminator

    classifier.load_state_dict(torch.load('result_model/'+ args.FL_algorithm +'/Jiawei_FedAvg_66.95_epoch_1001_dir_0.5_model_test_state_dict.pth'))    # load the target model
    evauator.load_state_dict(torch.load('result_model/'+ args.FL_algorithm +'/Jiawei_FedAvg_66.95_epoch_1001_dir_0.5_model_test_state_dict.pth'))        # load the evaluator
    # Generator.load_state_dict(torch.load('GAN_Generator.pkl'))         # load the generator
    # Disc.load_state_dict(torch.load('GAN_Discriminator.pkl'))       # load the discriminator

    labels = torch.LongTensor(args.batch_size).random_(0, 100).cuda()   # generate random labels

    inversion(args, classifier, evauator, Generator, Disc, labels)
