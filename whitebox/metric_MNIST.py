import torch
import numpy as np
from scipy.linalg import sqrtm
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import inception_v3
import torchvision.transforms as transforms
import torchvision
import torch
from pytorch_fid import fid_score
import pytorch_fid
from PIL import Image
from model import Inception_Net

def calculate_fid(real_images, generated_images, batch_size=64, device=torch.device('cpu')):
    # 加载Inception V3模型并将其设置为评估模式
    # inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    # inception_model.eval()
    inception_model = Inception_Net().to(device)     # load the inception model
    inception_model.load_state_dict(torch.load('Inception_Net.pth'))
    inception_model.eval()

    # 将数据集转换为张量并标准化为[-1, 1]范围内的值
    real_images = real_images.float().to(device)
    generated_images = generated_images.float().to(device)
    # real_images = (real_images - 0.5) * 2
    # generated_images = (generated_images - 0.5) * 2

    # 计算每个数据集的特征向量
    real_features = get_features(inception_model, real_images, batch_size, device)
    generated_features = get_features(inception_model, generated_images, batch_size, device)

    # 计算均值和协方差矩阵
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

    # 计算FID值
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

def get_features(model, images, batch_size, device):
    n_images = images.shape[0]
    n_batches = int(np.ceil(n_images / batch_size))
    features = np.zeros((n_images, 1024))
    for i in range(n_batches):
        batch_images = images[i*batch_size : (i+1)*batch_size]
        # print('batch_images shape: {}'.format(batch_images.shape))
        # print('batch_features shape: {}'.format(model(batch_images).shape))
        batch_features = model(batch_images, out_feature=True)
        batch_features = batch_features.view(batch_images.shape[0], -1).detach().cpu().numpy()
        features[i*batch_size : (i+1)*batch_size, ] = batch_features
    
    print('features shape: {}'.format(features.shape))
    print('features: {}'.format(features))
    for i in range(features.shape[1]):
        features[:, i] = (features[:, i] - np.min(features[:, i]))/(np.max(features[:, i]) - np.min(features[:, i]))

    print('features: {}'.format(features))

    return features

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # 计算均值差异和协方差矩阵平方根
    # mu1 = (mu1 - np.min(mu1))/(np.max(mu1) - np.min(mu1))
    # mu2 = (mu2 - np.min(mu2))/(np.max(mu2) - np.min(mu2))
    diff = mu1 - mu2
    print(diff.shape)
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # 计算FID值
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    mean_error = diff.dot(diff)
    print(mean_error)
    mean_error_2 = np.sum(diff**2)
    print(mean_error_2)
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * (covmean))

if __name__ == '__main__':
    # 加载数据集
    transform_train = transforms.Compose([
            # transforms.Resize([299,299]),
            # transforms.RandomCrop(299),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    transform_resize = transforms.Resize(28)
    # transform_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform_train)
    print(trainset.data.shape)
    train_data = trainset.data
    print(train_data[0].shape)
    train_data = torch.unsqueeze(train_data, 1)
    print(train_data.shape)
    

    

    # train_targets = trainset.targets
    # dict = {i: [] for i in range(10)}
    # idx = np.arange(len(trainset))      # index for train dataset
    # idxs_labels = np.vstack((idx, train_targets))      # index and label for train dataset
    # for index, label in zip(idxs_labels[0], idxs_labels[1]):
    #     dict[label.item()].append(train_data[index])
    # # select the images in order
    # selected_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 25 + [0, 1, 2, 3, 4, 5]
    # selected_data = []

    real_images = train_data
    print("The shape of one real images is: {}".format(real_images[0].shape))
    # real_images = np.transpose(real_images, (0, 3, 1, 2))
    generated_dataset = torch.load('FedDC/mnist/0.01_data.pth')         # load the FedDC generated dataset
    # generated_dataset = torch.load('fedmix/mnist/0.01_data.pth')         # load the FedMix generated dataset
    generated_images = generated_dataset['data']
    # generated_images = transform_normalize(generated_images)
    # generated_images = torch.load('fedmix/mnist/inversion_image/fake_image.pth')         # load the FedMix finetuned generated dataset
    generated_images = np.array([transform_resize(generated_images[i]).numpy() for i in range(len(generated_images))])
    # generated_images = np.transpose(generated_images, (0, 2, 3, 1))
    print('The shape of real images is: {}; the shape of generated image is: {}'.format(real_images.shape, generated_images.shape))

    real_images_1 = real_images[:1000]
    real_images_2 = real_images[1000:2000]
    # real_images = torch.from_numpy(real_images)
    generated_images = torch.from_numpy(generated_images)

    # noise as generated images
    generated_images = torch.randn(2048, 1, 28, 28)

    # 计算FID值
    fid_value = calculate_fid(real_images_1, generated_images, batch_size=1000, device=torch.device('cuda'))
    # fid_value = fid_value / (320 * real_images_1.shape[0])
    print('FID value is: {}'.format(fid_value))