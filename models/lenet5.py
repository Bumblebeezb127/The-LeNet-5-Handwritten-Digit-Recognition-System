"""
LeNet-5 卷积神经网络模型定义
基于Yann LeCun的经典LeNet-5架构，针对MNIST数据集(28x28灰度图)进行了适配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    LeNet-5模型结构

    架构说明:
    C1: 卷积层1 - 6个5x5卷积核 -> 6@24x24
    S2: 池化层1 - 2x2最大池化 -> 6@12x12
    C3: 卷积层2 - 16个5x5卷积核 -> 16@8x8
    S4: 池化层2 - 2x2最大池化 -> 16@4x4
    C5: 卷积层3 - 120个5x5卷积核 -> 120@1x1 (全连接)
    F6: 全连接层 - 120 -> 84
    F7: 输出层 - 84 -> 10

    原始LeNet-5设计用于32x32输入，MNIST为28x28，需要进行填充或调整
    """

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        self.fc1 = nn.Linear(120, 84)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_conv_layers(self):
        return [self.conv1, self.conv2, self.conv3]

    def get_feature_maps(self, x):
        features = []
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        features.append(x)
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        features.append(x)
        x = torch.relu(self.conv3(x))
        features.append(x)
        return features


class LeNet5Original(nn.Module):
    """
    原始LeNet-5结构（不含BatchNorm和Dropout）
    用于对比实验
    """

    def __init__(self, num_classes=10):
        super(LeNet5Original, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LeNet5()
    print(f"模型参数量: {count_parameters(model):,}")
    print(model)

    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    print(f"输出形状: {output.shape}")
