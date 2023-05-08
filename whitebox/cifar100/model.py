import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1), x # output both log_prob and unnormalized logits.

class Inception_Net(nn.Module):
    def __init__(self):
        super(Inception_Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 10, kernel_size=5),
                                   nn.BatchNorm2d(10))
        self.conv2 = nn.Sequential(nn.Conv2d(10, 20, kernel_size=5),
                                      nn.BatchNorm2d(20))
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Sequential(nn.Linear(320, 1024),
                                 nn.BatchNorm1d(1024))
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x, out_feature=False):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        if out_feature:
            return x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training)
        x = self.fc4(x)
        return F.log_softmax(x, dim = 1), x # output both log_prob and unnormalized logits.

class InverseMNISTNet(nn.Module):
    def __init__(self):
        super(InverseMNISTNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 320)
        self.deconv1 = nn.ConvTranspose2d(20, 10, kernel_size=5)
        self.deconv2 = nn.ConvTranspose2d(10, 1, kernel_size=5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 20, 4, 4)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.deconv1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.deconv2(x)
        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, input_size=32, n_class=10):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.sigmoid(self.fc4(x))

        return x

# define network for CIFAR10
class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1600):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2)))
        self.conv2 = nn.Sequential(nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2)))
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_feature=False):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        if return_feature:
            feature = out
        out = self.fc1(out)
        out = self.fc(out)

        if return_feature:
            return out, feature
        else:
            return F.log_softmax(out, dim = 1), out

class InverseCIFAR10Net(nn.Module):
    def __init__(self, out_features=3, z_dim=100, num_class=10, dim=1600):
        super(InverseCIFAR10Net, self).__init__()
        self.fc1 = nn.Linear(z_dim + num_class, 512)
        self.fc2 = nn.Linear(512, dim)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=5)
        self.deconv2 = nn.ConvTranspose2d(32, out_features, kernel_size=5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, 5, 5)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.deconv1(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.deconv2(x)
        return x

# define discriminator for CIFAR10
class CIFAR10Discriminator(nn.Module):
    def __init__(self, in_features=3, num_classes=1, dim=1600):
        super(CIFAR10Discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2)))
        self.conv2 = nn.Sequential(nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2)))
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128), 
            nn.ReLU(inplace=True)
            )
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

if __name__ == '__main__':
    x = torch.zeros((1,10))
    net = InverseMNISTNet()
    x = net(x)
    print(net.parameters)
    print(x.shape)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# add a linear layer to perform attack
class _ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear1 = nn.Linear(512*block.expansion, 256)
        self.linear2 = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


def ResNet(num, num_classes=10):
    if num == 18:
        return _ResNet(BasicBlock, [2,2,2,2], num_classes)    # change here to fit frozen neurons
    elif num == 34:
        return _ResNet(BasicBlock, [3,4,6,3], num_classes)
    elif num == 50:
        return _ResNet(Bottleneck, [3,4,6,3], num_classes)
    elif num == 101:
        return _ResNet(Bottleneck, [3,4,23,3], num_classes)
    elif num == 152:
        return _ResNet(Bottleneck, [3,8,36,3], num_classes)
    else:
        raise NotImplementedError


# A special framework for ResNet
class Sub_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(Sub_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # split the linear layer into two parts
        # self.linear1 = nn.Linear(512*block.expansion, 256)
        self.linear1_1 = nn.Linear(448*block.expansion, 224)
        self.linear1_2 = nn.Linear(64*block.expansion, 32)
        self.linear2 = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        # split the linear layer into two parts
        out_1, out_2 = out.split(448, 1)
        out_1 = self.linear1_1(out_1)
        out_2 = self.linear1_2(out_2)
        out = torch.cat([out_1, out_2], 1)
        # out = self.linear1(out)
        out = self.linear2(out)
        return out


# def ResNet(num, num_classes=10):
#     if num == 18:
#         return _ResNet(BasicBlock, [2,2,2,2], num_classes)
#     elif num == 34:
#         return _ResNet(BasicBlock, [3,4,6,3], num_classes)
#     elif num == 50:
#         return _ResNet(Bottleneck, [3,4,6,3], num_classes)
#     elif num == 101:
#         return _ResNet(Bottleneck, [3,4,23,3], num_classes)
#     elif num == 152:
#         return _ResNet(Bottleneck, [3,8,36,3], num_classes)
#     else:
#         raise NotImplementedError


# Add VGG neural network to deploy the subnet replacement attack
'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
# import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def partial_forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        partial_classifier = self.classifier[:-1]
        x = partial_classifier(x)
        return x



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))

