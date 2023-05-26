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
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from Model import MNIST_Net, InverseMNISTNet, Discriminator, FedAvgCNN, InverseCIFAR10Net

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


def inversion(args, classifier, evaluator, Generator, Disc, labels):

    gt_labels = torch.zeros(args.batch_size, 10).cuda()
    for i in range(args.batch_size):
        gt_labels[i][labels[i]] = 1       # generate one-hot labels
    print('The ground truth labels are: {}'.format(gt_labels))
    z = torch.randn(args.batch_size, args.z_dim).cuda()
    z = Variable(z, requires_grad=True).cuda()
    z = torch.concat((z, gt_labels), dim=1)     # concatenate the one-hot labels to z
    z = Variable(z, requires_grad=True).cuda()

    fake_image = Generator(z)
    # check if the directory exists
    if args.FL_algorithm == 'FedAvgDP' or args.FL_algorithm == 'FedPAQDP':
        if not os.path.exists('result/'+args.FL_algorithm+'/inversion_image_'+ args.dp_level):
            os.makedirs('result/'+args.FL_algorithm+'/inversion_image_'+args.dp_level)
        data_path = 'result/'+args.FL_algorithm+'/inversion_image_'+args.dp_level
    elif args.FL_algorithm == 'FedAvg' or args.FL_algorithm == 'FedProx' or args.FL_algorithm == 'FedPAQ':
        if args.local != True:
            if not os.path.exists('result/'+args.FL_algorithm+'/inversion_image_'+ args.acc):
                os.makedirs('result/'+args.FL_algorithm+'/inversion_image_'+args.acc)
            data_path = 'result/'+args.FL_algorithm+'/inversion_image_'+args.acc
        elif args.local == True:
            if not os.path.exists('result/'+args.FL_algorithm+'/inversion_image_local_'+args.acc):
                os.makedirs('result/'+args.FL_algorithm+'/inversion_image_local_'+args.acc)
            data_path = 'result/'+args.FL_algorithm+'/inversion_image_local_'+args.acc
    if args.seed != 2023:
        data_path = data_path + '_seed_' + str(args.seed)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
    # save the original image
    original_image = fake_image.clone().detach()
    tvls.save_image(original_image, data_path+'/original_image.png', normalize=False, range=(-1, 1))

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
    tvls.save_image(fake_image, data_path + '/fake_image.png', normalize=False, range=(-1, 1))
    # save the images and label as pth file
    torch.save(fake_image, data_path+'/fake_image.pth')

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

# test the model
def test(args, model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs = data[0].to(device)
            targets = data[1].to(device)
            _, outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Test dataset: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss / len(test_loader), correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--num_epoch', type=int, default=5000)
    parser.add_argument('--lambda_gt', type=int, default=1e7)
    parser.add_argument('--lambda_TV', type=int, default=1)
    parser.add_argument('--lambda_L2', type=int, default=1)
    parser.add_argument('--FL_algorithm', type=str, default='FedAvg')
    parser.add_argument('--dp_level', type=str, default='0.01')
    parser.add_argument('--acc', type = str, default='66.81')
    parser.add_argument('--local', action='store_true', default='use the local model or not')   
    parser.add_argument('--seed', type=int, default=2023, help='random seed (default: 1)')
    args = parser.parse_args()

    same_seeds(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('the arguments are: {}'.format(args))

    classifier = FedAvgCNN(in_features=3, dim=1600).to(device)      # target model
    evauator = FedAvgCNN(in_features=3, dim=1600).cuda()        # evaluator: evaluate the quality of the generated images, a model with high accuracy
    Generator = InverseCIFAR10Net().cuda()       
    Disc = Discriminator(input_size=3*32*32, n_class=1).cuda() 

    file_lis = os.listdir('result_model/'+ args.FL_algorithm)
    print('The file list is: {}'.format(file_lis))
    if args.FL_algorithm == 'FedAvgDP' or args.FL_algorithm == 'FedPAQDP':
        for file in file_lis:
            if args.dp_level in file and 'state_dict' in file:
                model_path = 'result_model/'+ args.FL_algorithm +'/'+ file
        print('The model path is: {}'.format(model_path))
        classifier.load_state_dict(torch.load(model_path))   # load the target model
        evauator.load_state_dict(torch.load(model_path))        # load the evaluator
    elif args.FL_algorithm == 'FedAvg' or args.FL_algorithm == 'FedProx' or args.FL_algorithm == 'FedPAQ':
        for file in file_lis:
            if args.acc in file and args.local != True:
                model_path = 'result_model/'+ args.FL_algorithm +'/'+ file
                break
            elif 'local' in file and args.local == True and args.acc in file:
                model_path = 'result_model/'+ args.FL_algorithm +'/'+ file
                break      
        print('The model path is: {}'.format(model_path))
        classifier = torch.load(model_path, map_location=device)
        evaluator = torch.load(model_path, map_location=device)

    # test the accuracy of the target model
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465,), (0.2023, 0.1994, 0.2010,))
            ])
    test_dataset = datasets.CIFAR10(root='../dataset', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=0)
    target_acc = test(args, classifier, test_loader, device)
    print('The accuracy of the target model is: {}'.format(target_acc))

    labels = torch.LongTensor(args.batch_size).random_(0, 10).cuda()   # generate random labels

    inversion(args, classifier, evauator, Generator, Disc, labels)

