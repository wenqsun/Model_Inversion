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

import torch.nn as nn

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


# network for CIFAR10
class FedAvgCNN(nn.Module):
    def __init__(self, in_features=3, num_classes=10, dim=1600):
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


if __name__ == '__main__':
    x = torch.zeros((1,10))
    net = InverseMNISTNet()
    x = net(x)
    print(net.parameters)
    print(x.shape)

'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, track=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes,track_running_stats=track)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes,track_running_stats=track)

        # self.bn1 = nn.GroupNorm(1, in_planes)
        # self.bn2 = nn.GroupNorm(1, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes,track_running_stats=track)
                # nn.GroupNorm(32,self.expansion*planes),
                # nn.GroupNorm(16,self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.relu(self.gn3( self.gn1(self.conv1(x))))
        
        # out = torch.nn.ReLU()(self.conv1(x))
        out = self.bn2(self.conv2(out))
        # out = self.gn4(self.gn2(self.conv2(out)))
        # out = self.conv2(out)
        a = self.shortcut(x) + out
        # out = F.relu(out)
        
        a = torch.nn.ReLU()(a)
        return a


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, track=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64,track_running_stats=track)
        self.latent_feature = int(512*block.expansion)
            # self.latent_feature = 25088
            # self.latent_feature = 2048
            # print(self.latent_feature)
        # self.gn1 = nn.GroupNorm(32,64)
        # self.gn2 = nn.GroupNorm(16,64)
        
        # # self.network._forward_impl = partial(forward, self.network)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, track=track)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, track=track)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, track=track)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, track=track)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(self.latent_feature, num_classes)  ####跟图片size有关，128=4*32
        # self.linear_1 = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, track=True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, track))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False):
        result = {}
        # out =torch.nn.ReLU()(self.conv1(x))
        
        # out = F.relu(self.gn2( self.gn1(self.conv1(x))))
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.avg_pool2d(out, 2)
        # out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        # out = self.avgpool(out)
        # out = out.view(out.size(0),out.size(1), -1)   ####和我们的方法维度对应起来
        # out = F.avg_pool1d(out, 2)
        feature = out.view(out.size(0), -1)
        # result["logit"] = out
        out = self.linear(feature)
        
        # out = self.linear_1(out)
        # result['output'] = out
        # return result
        # return F.log_softmax(out, dim = 1), None
        if return_feature:
            return out, feature
        else:
            return F.log_softmax(out, dim = 1), out


def resnet18( num_classes=100, track=True):
    return ResNet( BasicBlock, [2, 2, 2, 2],num_classes, track)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3],num_classes)


def ResNet50(num_classes=100):
    return ResNet(Bottleneck, [3, 4, 6, 3],num_classes)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# def test():
#     net = ResNet18()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())

# test()
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

