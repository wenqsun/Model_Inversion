import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import inception_v3
from scipy import linalg
import numpy as np
from PIL import Image
import torchvision.utils as tvls
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):#idxs is the index of dataset samples
        self.dataset = dataset
        self.idxs = list(idxs)
        #print(idxs,len(self.idxs))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# Define a function to calculate the FID score
def calculate_fid_score(real_images, generated_images, batch_size=64):
    # Load Inception-v3 network and set to evaluation mode
    inception_model = inception_v3(pretrained=True).cuda()
    inception_model.eval()
    inception_model.fc = nn.Identity()

    # Calculate activations for real images
    real_activations = []
    for i in range(0, len(real_images), batch_size):
        batch = real_images[i:i+batch_size].cuda()
        batch_activations = inception_model(batch).view(batch.size(0), -1)
        real_activations.append(batch_activations.cpu().detach())
    real_activations = torch.cat(real_activations, dim=0)

    # Calculate activations for generated images
    generated_activations = []
    for i in range(0, len(generated_images), batch_size):
        if i+batch_size > len(generated_images):
            break
        batch = generated_images[i:i+batch_size].cuda()
        batch_activations = inception_model(batch).view(batch.size(0), -1)
        generated_activations.append(batch_activations.cpu().detach())
    generated_activations = torch.cat(generated_activations, dim=0)

    # Calculate mean and covariance of real and generated activations
    mu1, sigma1 = real_activations.mean(dim=0), torch_cov(real_activations, rowvar=False)
    mu2, sigma2 = generated_activations.mean(dim=0), torch_cov(generated_activations, rowvar=False)

    # Calculate the squared difference between means and matrix trace of the product of covariances
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_score = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid_score

# Define a function to calculate covariance matrix using PyTorch
def torch_cov(m, rowvar=False):
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar:
        m = m.t()
    factor = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return factor * m.matmul(mt).squeeze()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='FedDC', help='FL method')
    parser.add_argument('--dp_level', type=float, default=0.01, help='DP level')
    parser.add_argument('--acc', type=str, default=66.81, help='accuracy of target model')
    parser.add_argument('--gpu_id', type=int, default=6, help='GPU ID')
    parser.add_argument('--local', action='store_true', help='use of local client model')
    parser.add_argument('--local_test', action='store_true', help='use of local test data')
    args = parser.parse_args()

    # Set GPU ID
    # os.CUDA_VISIBLE_DEVICES = '6'
     

    batch_size = 10000
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465,), (0.2023, 0.1994, 0.2010,))
    ])
    transform_resize = transforms.Resize((299, 299))
    dataset = datasets.CIFAR10(root="../dataset", train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print('load data done')
    # print(dataloader.dataset.data.shape)
    # load the first batch of real images
    for i, (real_images, _) in enumerate(dataloader):
        if i == 0:
            print(real_images.shape)
            break
    real_images = iter(dataloader).__next__()[0]
    print('get real data ready')
    if args.local_test == True:
        real_images = torch.load(f'result/{args.method}/LocalData.pth')
        local_dataloader = DataLoader(real_images, batch_size=5000, shuffle=True)
        real_images = iter(local_dataloader).next()[0]
        print(f'the shape is {real_images.shape}')
        real_images = np.array([transform_resize(real_images[i]).numpy() for i in range(real_images.shape[0])])
        real_images = torch.from_numpy(real_images).cpu().detach()
    print(f"Real images: {real_images.shape}")

    # Load generated images
    if args.method == 'FedDC' or args.method == 'FedMix' or args.method == 'FedReal':
        fake_images = torch.load(f"./{args.method}/0.1_data.pth")
        fake_images = fake_images['data']
        # resize fake images
        fake_images = np.array([transform_resize(fake_images[i]).numpy() for i in range(fake_images.shape[0])])
        fake_images = torch.from_numpy(fake_images)
        print(f"Fake images: {fake_images.shape}")
    elif args.method == 'FedAvg' or args.method == 'FedProx' or args.method == 'FedPAQ' or args.method == 'FedMD' or args.method == 'FedED':
        if args.local == False:
            fake_images = torch.load(f'result/{args.method}/inversion_image_{args.acc}/fake_image.pth').cpu().detach()
        elif args.local == True:
            fake_images = torch.load(f'result/{args.method}/inversion_image_local_{args.acc}/fake_image.pth').cpu().detach()
        # resize fake images
        fake_images = np.array([transform_resize(fake_images[i]).numpy() for i in range(fake_images.shape[0])])
        fake_images = torch.from_numpy(fake_images)
        print(f"Fake images: {fake_images.shape}")
    elif args.method == 'FedAvgDP' or args.method == 'FedMDDP' or args.method == 'FedEDDP' or args.method == 'FedPAQDP':
        fake_images = torch.load(f'result/{args.method}/inversion_image_{args.dp_level}/fake_image.pth').cpu().detach()
        # resize fake images
        fake_images = np.array([transform_resize(fake_images[i]).numpy() for i in range(fake_images.shape[0])])
        fake_images = torch.from_numpy(fake_images)
        print(f"Fake images: {fake_images.shape}")
    elif args.method == 'noise':
        fake_images = torch.randn(10000, 3, 299, 299)

    # Calculate FID score
    fid_score = calculate_fid_score(real_images, fake_images, batch_size=32)
    print(f"FID score: {fid_score:.2f}")