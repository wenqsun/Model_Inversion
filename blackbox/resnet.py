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


if __name__ == "__main__":

    model = resnet18()
    print(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(pytorch_total_params)

    print
# def test():
#     net = ResNet18()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())

# test()