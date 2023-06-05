import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from scipy import linalg
import numpy as np
from PIL import Image
import torchvision.utils as tvls
import argparse
import os
import warnings
warnings.filterwarnings("ignore")
import math

# compute PSNR for three channels
def PSNR(img1, img2):
    # Convert images to float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # Compute MSE (Mean Squared Error) for each channel
    mse_r = np.mean((img1[0, :,:] - img2[0, :,:]) ** 2)
    mse_g = np.mean((img1[1, :,:] - img2[1, :,:]) ** 2)
    mse_b = np.mean((img1[2, :,:] - img2[2, :,:]) ** 2)
    
    # Compute average MSE across all channels
    mse = (mse_r + mse_g + mse_b) / 3.0
    
    # Compute PSNR using the formula: PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    max_i = 1   # maximum pixel intensity for 8-bit images
    psnr = 20 * np.log10(max_i) - 10 * np.log10(mse)
    
    return psnr

def compute_PSNR(real_images, generated_images):
    best_psnr_list = []
    for i in range(generated_images.shape[0]):
        best_psnr = 0
        for j in range(real_images.shape[0]):
            if PSNR(real_images[j], generated_images[i]) > best_psnr:
                best_psnr = PSNR(real_images[j], generated_images[i])
                best_index = j
        
        best_psnr_list.append(best_psnr)
        # print('best_psnr: ', best_psnr)

    return np.mean(best_psnr_list)

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
    parser.add_argument('--gpu_id', type=int, default=6, help='GPU ID')
    parser.add_argument('--acc', type=float, default=0.5, help='accuracy of the model')
    args = parser.parse_args()

    # Set GPU ID
    # os.CUDA_VISIBLE_DEVICES = '6'
     

    batch_size = 10000
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343,), (0.2682515741720801, 0.2573637364478126, 0.2770957707973042))
    ])
    transform_resize = transforms.Resize((299, 299))
    dataset = datasets.CIFAR100(root="../dataset", train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    real_images = iter(dataloader).next()[0]
    print(f"Real images: {real_images.shape}")

    # Load generated images
    if args.method == 'FedDC' or args.method == 'FedMix' or args.method == 'FedReal':
        fake_images = torch.load(f"./{args.method}/0.5_data.pth")
        fake_images = fake_images['data']
        # resize fake images
        fake_images = np.array([transform_resize(fake_images[i]).numpy() for i in range(fake_images.shape[0])])
        fake_images = torch.from_numpy(fake_images)
        print(f"Fake images: {fake_images.shape}")
    elif args.method == 'FedAvg' or args.method == 'FedProx' or args.method == 'FedPAQ' or args.method == 'FedMD' or args.method == 'FedED':
        fake_images = torch.load(f'result/{args.method}/inversion_image_{args.acc}/generate_image.pth').cpu().detach()
        # resize fake images
        fake_images = np.array([transform_resize(fake_images[i]).numpy() for i in range(fake_images.shape[0])])
        fake_images = torch.from_numpy(fake_images)
        print(f"Fake images: {fake_images.shape}")
    elif args.method == 'noise':
        fake_images = torch.randn(10000, 3, 299, 299)

    # Calculate FID score
    fid_score = calculate_fid_score(real_images, fake_images, batch_size=32)
    print(f"FID score: {fid_score:.2f}")