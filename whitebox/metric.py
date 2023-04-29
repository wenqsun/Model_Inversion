import numpy as np
import math
import torch
import torchvision.transforms as transforms
import torchvision
import argparse
from PIL import Image

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
    parser.add_argument('--data_path', type=str, default='../data', metavar='N', help='dataset path')
    parser.add_argument('--method', type=str, default='FedDC', metavar='N', help='model name')
    args = parser.parse_args()
    transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
    original_dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform_train)
    original = original_dataset.data.numpy()
    print(original[0])
    # original = np.array([[0, 0, 0], [0, 0, 0]])
    # compressed = np.array([[0, 0, 0], [0, 0, 0]])
    if args.method == 'FedDC':
        compressed = torch.load('FedDC/mnist/0.01_data.pth')['data'].numpy()        # load the FedDC generated dataset
    elif args.method == 'FedMix_original':
        compressed = torch.load('fedmix/mnist/0.01_data.pth')['data'].numpy()        # load the FedMix generated dataset
    elif args.method == 'FedMix_finetuned':
        compressed = torch.load('fedmix/mnist/inversion_image/fake_image.pth').numpy()        # load the FedMix finetuned generated dataset
    elif args.method == 'FedAvg':
        compressed = torch.load('result/inversion_image/fake_image.pth').cpu().detach().numpy()        # load the FedAvg generated dataset
    compressed = compressed.reshape(-1, 28, 28)
    original = original[:1000]
    compressed = compressed[:1000]
    print(compressed[0])
    # *255 for each image
    for i in range(compressed.shape[0]):
        for j in range(compressed.shape[1]):
            for k in range(compressed.shape[2]):
                compressed[i][j][k] = compressed[i][j][k] * 255
                if compressed[i][j][k] > 255:
                    compressed[i][j][k] = 255
                elif compressed[i][j][k] < 0:
                    compressed[i][j][k] = 0

    # search the best image from the compressed for the original image
    best_psnr_list = []
    for i in range(compressed.shape[0]):
        best_psnr = 0
        for j in range(original.shape[0]):
            if PSNR(original[j], compressed[i]) > best_psnr:
                best_psnr = PSNR(original[j], compressed[i])
                best_index = j
        
        best_psnr_list.append(best_psnr)

    print('the best psnr is: {}'.format((best_psnr_list)))
    print('the mean largeset psnr is: {}'.format(np.mean(best_psnr_list)))   
    

    # for i in range(10):
    #     compressed[i] = Image.fromarray(compressed[i].astype(np.uint8))
    # save the generated images

    # img = Image.fromarray(compressed[0].astype('uint8'))
    # if args.method == 'FedDC':
    #     img.save('result/metric/FedDC_0.01_data.jpg')
    # elif args.method == 'FedMix_original':
    #     img.save('result/metric/FedMix_original_0.01_data.jpg')
    # elif args.method == 'FedMix_finetuned':
    #     img.save('result/metric/FedMix_finetuned_0.01_data.jpg')
    # print(compressed[0])
    # psnr = PSNR(original[0], compressed[0])
    # print(psnr)