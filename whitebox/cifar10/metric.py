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
    batch_size = 64
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_resize = transforms.Resize((299, 299))
    train_dataset = datasets.CIFAR10(root='../dataset', train=True, download=True, transform=transform)
    print('shape of training data:', train_dataset.data.shape)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    x = train_dataset.data[0].transpose((2, 0, 1))
    print('the shape of x:', x.shape)
    x = transform_resize(torch.from_numpy(x))
    print('the shape of x:', x.shape)
    

    # Generate fake images using your model of choice
    # (replace this with your own code to generate fake images)
    fake_images = torch.randn(10000, 3, 32, 32)
    fake_images = np.array([transform_resize(fake_images[i]).numpy() for i in range(len(fake_images))])
    fake_images = torch.from_numpy(fake_images).float()

    # Calculate FID score
    real_images = train_loader.dataset.data.transpose((0, 3, 1, 2))
    real_images = np.array([transform_resize(torch.from_numpy(real_images[i])).numpy() for i in range(len(real_images))])
    real_images = torch.from_numpy(real_images).float()
    fake_images = real_images[10000:20000]
    real_images = real_images[:100]
    print('the shape of real_images:', real_images.shape)
    # real_images = torch.randn(10000, 3, 299, 299)
    fid_score = calculate_fid_score(real_images, fake_images, batch_size=32)
    print(f"FID score: {fid_score:.2f}")