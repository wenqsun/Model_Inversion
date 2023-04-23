from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os, shutil
from torchvision import datasets, transforms
from model import MNIST_Net, InverseMNISTNet
import torch.nn.functional as F
import torchvision.utils as vutils

def train(classifier, generator, log_interval, device, data_loader, optimizer, epoch):
    classifier.eval()
    generator.train()

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            prediction, _ = classifier(data)
        # print(prediction)
        reconstruction = generator(prediction)
        loss = F.mse_loss(reconstruction, data)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))

def test(classifier, generator, device, data_loader, test_type, epoch):
    classifier.eval()
    generator.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            prediction, _ = classifier(data)
            reconstruction = generator(prediction)
            test_loss += F.mse_loss(reconstruction, data, reduction='sum').item()  # sum up batch loss
    
    for i in range(20):
        vutils.save_image(reconstruction[i], 'result/test/reconstruction_image_{}.png'.format(i))

    test_loss /= len(data_loader.dataset)

    print('\n Train Epoch: {}  Test inversion model on {} set: Average MSE loss: {:.6f}\n'.format(epoch, test_type, test_loss))
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',help='device to use (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

    classifier = MNIST_Net().to(device)
    classifier.load_state_dict(torch.load('MNIST_Net.pth'))
    generator = InverseMNISTNet().to(device)
    optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)


    for epoch in range(1, args.epochs + 1):
        train(classifier, generator, args.log_interval, device, train_dataloader, optimizer, epoch)
        test(classifier, generator, device, test_dataloader, 'test', epoch)
    if args.save_model:
            torch.save(generator.state_dict(), "Generator_MNIST.pth")

    # Get the reconstruction result for each label data
    for i in range(10):
        label = torch.zeros(1, 10).to(device)
        label[0][i] = 1
        reconstruction_image = generator(label)
        vutils.save_image(reconstruction_image, 'result/reconstruction_image_{}.png'.format(i))