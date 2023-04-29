# set the random seed
import random
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable
from model import MNIST_Net, Inception_Net


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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_file = datasets.MNIST(
    root='../dataset/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_file = datasets.MNIST(
    root='../dataset/',
    train=False,
    transform=transforms.ToTensor()
)

EPOCH = 10
BATCH_SIZE = 128
LR = 1E-3

train_loader = DataLoader(
    dataset=train_file,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_file,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = Inception_Net().to(device)
optim = torch.optim.Adam(model.parameters(), LR)
lossf = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (data, targets) in enumerate(train_loader):
        optim.zero_grad()
        data = data.to(device)
        targets = targets.to(device)
        _, output = model(data)
        loss = lossf(output, targets)
        acc = (output.argmax(1) == targets).sum().item()/data.size()[0]
        loss.backward()
        optim.step()
        if step % 200 == 0:
            for image, label in test_loader:
                image = image.to(device)
                label = label.to(device)
                _, output = model(image)
                test_loss = lossf(output, label)
                test_acc = (output.argmax(1) == label).sum().item()/image.size()[0]
            print(
                f'EPOCH: {epoch+1:0>{len(str(EPOCH))}}/{EPOCH}',
                f'STEP: {step+1:0>{len(str(len(train_loader)))}}/{len(train_loader)}',
                f'Training_LOSS: {loss.item():.4f}',
                f'Training_ACC: {acc:.4f}',
                f'Testing_LOSS: {test_loss.item():.4f}',
                f'Testing_ACC: {test_acc:.4f}',
            )

torch.save(model.state_dict(), 'Inception_Net.pth')