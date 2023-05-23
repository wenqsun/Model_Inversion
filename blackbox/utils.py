from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os, shutil
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.utils as tvls


class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, data=None, batch_size=64):
        self.images = data
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)


# test the model
def test(args, model, test_loader, epoch, device):
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
            # if batch_idx % args.log_interval == 0:
            #     print('Test Batch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
            #         batch_idx, batch_idx * len(inputs), len(test_loader.dataset),
            #         100. * batch_idx / len(test_loader), loss.item(), 100. * correct / total))

    print('Epoch: {} || Test dataset: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, test_loss / len(test_loader), correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    print('done')
# def train(classifier, generator, log_interval, device, data_loader, optimizer, epoch):
#     classifier.eval()
#     generator.train()

#     for batch_idx, (data, target) in enumerate(data_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         with torch.no_grad():
#             prediction, _ = classifier(data)
#         # print(prediction)
#         reconstruction = generator(prediction)
#         loss = F.mse_loss(reconstruction, data)
#         loss.backward()
#         optimizer.step()

#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(data_loader.dataset),
#                 100. * batch_idx / len(data_loader), loss.item()))

# def test(classifier, generator, device, data_loader, test_type, epoch):
#     classifier.eval()
#     generator.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for data, target in data_loader:
#             data, target = data.to(device), target.to(device)
#             prediction, _ = classifier(data)
#             reconstruction = generator(prediction)
#             test_loss += F.mse_loss(reconstruction, data, reduction='sum').item()  # sum up batch loss
    
#     for i in range(20):
#         vutils.save_image(reconstruction[i], 'result/test/reconstruction_image_{}.png'.format(i))

#     test_loss /= len(data_loader.dataset)

#     print('\n Train Epoch: {}  Test inversion model on {} set: Average MSE loss: {:.6f}\n'.format(epoch, test_type, test_loss))