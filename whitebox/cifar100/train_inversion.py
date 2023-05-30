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
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.datasets as datasets

from Model import MNIST_Net, InverseMNISTNet, Discriminator, FedAvgCNN, InverseCIFAR10Net, InverseCIFAR100Net
from resnet import resnet18

from metric import calculate_fid_score, torch_cov

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

def TV_prior(image):
    # compute the total variation prior
    # image: [batch_size, 3, 32, 32]
    diff1 = image[:,:,:,:-1] - image[:,:,:,1:]
    diff2 = image[:,:,:-1,:] - image[:,:,1:,:]
    diff3 = image[:,:,1:,:-1] - image[:,:,:-1,1:]
    diff4 = image[:,:,:-1,:-1] - image[:,:,1:,1:]

    return torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

def test(args, model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    label_list = [0 for i in range(args.num_class)]
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs = data[0].to(device)
            targets = data[1].to(device)
            _, outputs = model(inputs)
            # print('the shape of inputs, targets, outputs are: {}, {}, {}'.format(inputs.shape, targets.shape, outputs.shape))
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            for i in range(targets.size(0)):
                label_list[targets[i].item()] += 1
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('Test dataset: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss / len(test_loader), correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset), label_list

# build a dataset for generated dataset with labels
class GeneratedDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.label[idx]
        if self.transform:
            image = self.transform(image)

        return image, label


def inversion(args, classifier, evaluator, Generator, Disc, labels, transform_resize, target_acc, device):

    gt_targets = torch.zeros(args.batch_size, 100).cuda()
    for i in range(args.batch_size):
        gt_targets[i][labels[i]] = 1       # generate one-hot labels
    print('The ground truth labels are: {}'.format(gt_targets))

    z = torch.randn(args.batch_size, args.z_dim).cuda()
    z = Variable(z, requires_grad=True).cuda()
    z = torch.concat((z, gt_targets), dim=1)     # concatenate the one-hot labels to z
    fake_image = Generator(z)
    # save the original image
    original_image = fake_image.clone().detach()
    data_path = 'result/'+args.FL_algorithm+'/inversion_image_'+args.acc+'_epoch_'+str(args.num_epoch)+'_use_DenseGen_'+str(args.use_dense)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    tvls.save_image(original_image, data_path + '/original_image.png', normalize=False, range=(-1, 1))

    # log the loss
    log = open(data_path + '/log.txt', 'w')
    log.write('epoch, loss, gt_loss, prior_loss, TV_loss, L2_loss\n')
    log.flush()

    # log the fid score
    real_image = torch.load('result/real_images.pth')      # get the real images to compute the fid score
    fid_list = []
    attack_acc_list = []
    generated_label_list = []
    best_loss = 10000000

    optimizer = optim.Adam(Generator.parameters(), lr=args.lr, betas=(0.5, 0.999))      # optimizer for generator
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, (args.num_epoch+1)):
        # generate the noise z
        z = torch.randn(args.batch_size, args.z_dim).cuda()
        z = Variable(z, requires_grad=True).cuda()
        labels = torch.LongTensor(args.batch_size).random_(0, 100).cuda()
        gt_targets = torch.zeros(args.batch_size, 100).cuda()
        for i in range(args.batch_size):
            gt_targets[i][labels[i]] = 1       # generate one-hot labels
        z = torch.concat((z, gt_targets), dim=1)     # concatenate the one-hot labels to z

        optimizer.zero_grad()
        fake_image = Generator(z)
        _, fake_out = classifier(fake_image)

        D_fake = Disc(fake_image.view(-1, 3*32*32))     # prior result for discriminator
        prior_loss = -torch.mean(D_fake)    # prior loss
        gt_loss = criterion(fake_out, gt_targets)     # gt loss
        TV_loss = TV_prior(fake_image)      # total variation prior
        L2_loss = torch.norm(fake_image, 2)     # L2 loss
        loss = 0 * prior_loss + args.lambda_gt * gt_loss + args.lambda_TV * TV_loss + args.lambda_L2 * L2_loss
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print('epoch: {}, loss: {}, gt_loss: {}, prior_loss: {}, TV_loss: {}, L2_loss: {}'.format(epoch, loss.item(), gt_loss.item(), prior_loss.item(), TV_loss.item(), L2_loss.item()))
            # log the loss
            log.write('{},\t{},\t{},\t{},\t{},{}\n'.format(epoch, loss.item(), gt_loss.item(), prior_loss.item(), TV_loss.item(), L2_loss.item()))
            log.flush()
        if epoch % args.log_interval == 0:
            # compute the fid score every 500 epoch
            z = torch.randn(args.data_num, args.z_dim).cuda()
            z = Variable(z, requires_grad=True).cuda()
            labels = torch.LongTensor(args.data_num).random_(0, 100).cuda()
            gt_labels = torch.zeros(args.data_num, 100).cuda()
            for i in range(args.data_num):
                gt_labels[i][labels[i]] = 1       # generate one-hot labels
            z = torch.concat((z, gt_labels), dim=1)     # concatenate the one-hot labels to z
            fake_image = Generator(z).cpu().detach()
            fake_image_original = fake_image.clone()
            # resize the images to 299*299
            fake_image = np.array([transform_resize(fake_image[i]).numpy() for i in range(fake_image.shape[0])])
            fake_image = torch.from_numpy(fake_image)
            print('The shape of the fake image is: {}'.format(fake_image.shape))
            fid_score = calculate_fid_score(real_image, fake_image, batch_size=32)
            print('The fid score is: {}'.format(fid_score))
            fid_list.append(fid_score)

            # test the inference quality of generated images
            # test the inference quality of generated images
            dataset = GeneratedDataset(fake_image_original, labels)
            dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
            attack_acc, label_list = test(args, evaluator, dataloader, device)
            print('The attack accuracy is: {}'.format(attack_acc))
            print('The label list is: {}'.format(label_list))
            generated_label_list.append(label_list)
            attack_acc_list.append(attack_acc)

        if loss < best_loss:
            best_loss = loss
            torch.save(Generator, data_path + '/Generator.pth')
            fake_image = Generator(z).cpu().detach()
            tvls.save_image(fake_image, data_path + '/generate_image.png', normalize=False, range=(-1, 1))
            torch.save(fake_image, data_path + '/generate_image.pth')
    
    # read loss from the txt, and plot the loss curve
    loss_path = data_path + '/loss.png'
    loss = np.loadtxt(data_path + '/log.txt', skiprows=1, delimiter=',')
    plt.figure()
    plt.plot(loss[:, 0], loss[:, 2], label='gt_loss')
    plt.xlabel('epoch')
    plt.ylabel('gt_loss')
    plt.savefig(loss_path)

    # plot the fid score curve
    fid_path = data_path + '/fid.png'
    plt.figure()
    plt.plot(np.arange(args.log_interval, (args.num_epoch+1), args.log_interval), fid_list)
    plt.xlabel('epoch')
    plt.ylabel('fid_score')
    plt.savefig(fid_path)

    # plot the attack accuracy curve
    attack_acc_path = data_path + '/attack_acc.png'
    plt.figure()
    plt.plot(np.arange(args.log_interval, (args.num_epoch+1), args.log_interval), attack_acc_list)
    plt.xlabel('epoch')
    plt.ylabel('attack_acc (%)')
    plt.savefig(attack_acc_path)
    
    # log the fid score and attack accuracy
    print('---------------------------------------------------------------------')
    print('The best FID score is: {}'.format(min(fid_list)))
    print('The best attack accuracy is: {}'.format(max(attack_acc_list)))
    log.write('The FID score is: {}\n'.format(fid_list))
    log.write('The best FID score is: {}\n'.format(min(fid_list)))
    log.write('The attack accuracy is: {}\n'.format(attack_acc_list))
    log.write('The best attack accuracy is: {}\n'.format(max(attack_acc_list)))
    log.write('The target model accuracy is {}\n'.format(target_acc))
    log.flush()
    log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--num_epoch', type=int, default=10000)
    parser.add_argument('--lambda_gt', type=int, default=1e7)
    parser.add_argument('--lambda_TV', type=int, default=1)
    parser.add_argument('--lambda_L2', type=int, default=1)
    parser.add_argument('--lambda_infor', type=int, default=1)
    parser.add_argument('--lambda_feature', type=int, default=1)
    parser.add_argument('--FL_algorithm', type=str, default='FedAvg')
    parser.add_argument('--acc', type=str, default='66.95')
    parser.add_argument('--data_num', type=int, default=10000)
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--dp_level', type=str, default='0.01')
    parser.add_argument('--num_class', type=int, default=100)
    parser.add_argument('--use_dense', action='store_true', help='use dense Generator')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    same_seeds(2023)

    classifier = resnet18().cuda()      # target model
    evauator = resnet18().cuda()      # evaluator: evaluate the quality of the generated images, a model with high accuracy
    if args.use_dense:
        Generator = InverseCIFAR100Net(z_dim = args.z_dim, num_class=100).cuda()         # generator
    else:
        Generator = InverseCIFAR10Net(z_dim = args.z_dim, num_class=100).cuda()         # generator
    Disc = Discriminator(input_size=3*32*32, n_class=1).cuda()       # discriminator

    file_lis = os.listdir('result_model/'+ args.FL_algorithm)
    print('The file list is: {}'.format(file_lis))
    for file in file_lis:
        if args.acc in file:
            model_path = 'result_model/'+ args.FL_algorithm + '/' + file
    print('The model path is: {}'.format(model_path))
    classifier = torch.load(model_path, map_location=device)    # load the target model
    evauator = torch.load('result_model/Central/epoch_1000_dir_0.5_model_test.pth')       # load the evaluator

    labels = torch.LongTensor(args.batch_size).random_(0, 100).cuda()   # generate random labels
    transform_resize = transforms.Resize((299, 299))   # resize the images to 299*299

    # test the target model
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343,), (0.2682515741720801, 0.2573637364478126, 0.2770957707973042),)
            ])
    test_loader = DataLoader(datasets.CIFAR100(root="../dataset", train=False, transform=test_transform, download=True), batch_size=1000, shuffle=False)
    target_acc, _ = test(args, classifier, test_loader, device)
    evauator_acc, _ = test(args, evauator, test_loader, device)
    print('The accuracy of evaluation model is: {}'.format(evauator_acc))


    inversion(args, classifier, evauator, Generator, Disc, labels, transform_resize, target_acc, device)
