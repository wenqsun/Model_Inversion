from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os, shutil
from torchvision import datasets, transforms
from Model import MNIST_Net, InverseMNISTNet, InverseCIFAR10Net, FedAvgCNN, resnet18
import torch.nn.functional as F
import torchvision.utils as tvls
from utils import UnlabeledDataset, test, reset_model


def generate_data(args, teacher, student, generator, labels, num_class, epoch, device):
    teacher.eval()
    student.eval()
    current_epoch = epoch
    best_loss = 1e6
    best_inputs = None
    z = torch.randn(args.batch_size, args.z_dim).to(device)
    z.requires_grad = True
    targets = torch.zeros(args.batch_size, num_class).to(device)
    for i in range(args.batch_size):
        targets[i][labels[i]] = 1       # one-hot encoding
    print('the ground truth label is {}'.format(targets))
    z = torch.concat((z, targets), dim=1)     # concatenate z and targets

    # reset_model(generator)
    optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    for epoch in range(args.generate_epochs):
        optimizer.zero_grad()
        inputs = generator(z)
        prob, s_out = student(inputs)
        ce_loss = F.cross_entropy(s_out, targets)

        # loss_infor = 0 * torch.mul(prob, torch.log(prob)).mean()
        loss_infor = torch.tensor(0)
        loss = ce_loss + loss_infor
        loss.backward()
        optimizer.step()

        # if epoch % args.log_interval == 0:
        print('Train Epoch: {} \tLoss: {}\t CrossEntropy Loss: {}\t Information Loss: {}'.format(epoch, loss.item(), ce_loss.item(), loss_infor.item()))

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_inputs = inputs
    # check id directory exists, if not, create it
    if os.path.exists('result/'+args.dataset+'/' + args.method) == False:
        os.makedirs('result/'+args.dataset+'/' + args.method)
    # save the best input
    tvls.save_image(best_inputs, 'result/'+args.dataset+'/' + args.method + '/best_input.png')

    return best_inputs, best_loss


def train_kd(args, teacher, student, optimizer, data_loader, epoch, device):       # train student model with knowledge distillation using generated dataset
    teacher.eval()
    student.train()
    current_epoch = epoch
    # optimizer = optim.Adam(student.parameters(), lr=args.lr, betas=(0.5, 0.999))
    for epoch in range(args.kd_epochs):
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
            data = data.detach()
            optimizer.zero_grad()
            with torch.no_grad():
                _, t_out = teacher(data)
            _, s_out = student(data)
            loss = abs(F.cross_entropy(s_out, t_out))
            loss.backward()
            optimizer.step()


        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tKD Loss: {:.6f}'.format(
                current_epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
        
    return loss.item()
          

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use (cifar10, cifar100)')
    parser.add_argument('--method', type = str, default='FedMD', help='method to use (FedMD, FedEG)')
    parser.add_argument('--z_dim', type=int, default=100, metavar='N', help='dimension of latent variable z (default: 100)')
    parser.add_argument('--batch-size', type=int, default=250, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--generate_batch_size', type=int, default=128, metavar='N', help='input batch size for generating (default: 64)')

    parser.add_argument('--train_epochs', type=int, default=3, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--generate_epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--kd_epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda',help='device to use (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    if args.dataset == 'cifar10':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465,), (0.2023, 0.1994, 0.2010,))
            ])
        train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('../data', train=False, download=True, transform=transform)

        num_class = 10
        student_net = FedAvgCNN().to(device)
    elif args.dataset == 'cifar100':
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343,), (0.2682515741720801, 0.2573637364478126, 0.2770957707973042),)
            ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343,), (0.2682515741720801, 0.2573637364478126, 0.2770957707973042),)
            ])
        train_dataset = datasets.CIFAR100('../data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100('../data', train=False, download=True, transform=test_transform)

        num_class = 100
        student_net = resnet18().to(device)     # class number is 100

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

    # teacher = FedAvgCNN().to(device)   # load the target model
    ensemble_weight = torch.load('result_model/'+ args.method + '/dir_0.1_model_test.pth', map_location=device)  # load the target model
    teacher = ensemble_weight[0]    # use the first model in the ensemble model as the target model
    torch.save(teacher.state_dict(), 'result_model/'+ args.method + '/teacher_model.pth')   # save the target model
    generator = InverseCIFAR10Net(num_class=num_class).to(device)
    labels = torch.LongTensor(args.batch_size).random_(0, num_class).cuda()   # generate random labels

    optimizer = optim.Adam(student_net.parameters(), lr=args.lr, betas=(0.5, 0.999))   # optimizer for the student model

    # test the accuracy of the classifier (target model)
    teacher_acc = test(args, teacher, test_dataloader, 0, device)

    # create a log file
    if os.path.exists('result/'+args.dataset+'/' + args.method) == False:
        os.makedirs('result/'+args.dataset+'/' + args.method)
    log_file = open('result/'+args.dataset+'/' + args.method + '/log.txt', 'w')
    log_file.write('Epoch:'+' '+'Generate_loss'+' '+'KD_loss'+' '+'Teacher_acc'+' '+'Student_acc'+'\n')

    for epoch in range(1, args.train_epochs + 1):
        gen_data, generate_loss = generate_data(args, teacher, student_net, generator, labels, num_class, epoch, device)
        data_loader = torch.utils.data.DataLoader(UnlabeledDataset(data=gen_data), batch_size=args.generate_batch_size, shuffle=True)    # dataloader for the generated data
        kd_loss = train_kd(args, teacher, student_net, optimizer, data_loader, epoch, device)
        student_acc = test(args, student_net, test_dataloader, epoch, device)
        # log the generate_loss, kd_loss, teacher_acc, student_acc
        log_file.write('Epoch:'+' '+str(epoch)+' || '+str(generate_loss) + ' ' + str(kd_loss) + ' ' + str(teacher_acc) + ' ' + str(student_acc) + '\n')
    
    # save the generated data
    torch.save(gen_data, 'result/'+args.dataset+'/' + args.method + '/gen_data.pth')