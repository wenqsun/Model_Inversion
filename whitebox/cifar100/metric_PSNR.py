import numpy as np
import math
import torch
import torchvision.transforms as transforms
import torchvision
import argparse
from PIL import Image
import torchvision.utils as tvls

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generated images quality evaluation')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='N', help='dataset name')
    parser.add_argument('--data_path', type=str, default='../dataset', metavar='N', help='dataset path')
    parser.add_argument('--method', type=str, default='FedDC', metavar='N', help='model name')
    args = parser.parse_args()
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343,), (0.2682515741720801, 0.2573637364478126, 0.2770957707973042))
        ])

    original_dataset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
    original_loader = torch.utils.data.DataLoader(original_dataset, batch_size=1000, shuffle=False, num_workers=2)
    original_data = iter(original_loader).next()[0].numpy()
    tvls.save_image(torch.from_numpy(original_data[0]).float(), 'original/original_images.png', nrow=1)
    print('the shape of original_data:', original_data.shape)
    # print('the value of original_data:', original_data[0])

    # load generated images
    if args.method == 'FedDC' or args.method == 'FedMix' or args.method == 'FedReal':
        generated_image = torch.load(args.method + '/0.5_data.pth')
        generated_image = generated_image['data'].numpy()
        tvls.save_image(torch.from_numpy(generated_image[1]).float(), args.method+'/fake_image.png', nrow=1)
    elif args.method == 'FedAvg' or args.method == 'FedProx':
        generated_image = torch.load(f'result/{args.method}/inversion_image/fake_image.pth').cpu().detach()
        generated_image = generated_image.numpy()
    elif args.method == 'noise':
        generated_image = torch.randn(1000, 3, 32, 32).numpy()

    # scale original images to [0, 255]
    original_data = original_data * 255
    original_data = np.clip(original_data, 0, 255) # clamp original images to [0, 255]
    print(original_data[0])

    # scale generated images to [0, 255]
    generated_image = generated_image * 255
    generated_image = np.clip(generated_image, 0, 255) # clamp generated images to [0, 255]
    print(generated_image[0])

    # search the best image from the compressed for the original image
    best_psnr_list = []
    for i in range(generated_image.shape[0]):
        best_psnr = 0
        for j in range(original_data.shape[0]):
            if PSNR(original_data[j], generated_image[i]) > best_psnr:
                best_psnr = PSNR(original_data[j], generated_image[i])
                best_index = j
        
        best_psnr_list.append(best_psnr)

    # print('the best psnr is: {}'.format((best_psnr_list[:10])))
    print('the mean largeset psnr is: {}'.format(np.mean(best_psnr_list)))  
