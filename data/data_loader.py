"""
数据加载和预处理模块
负责MNIST数据集的下载、加载、预处理以及数据增强
"""

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os


class MNISTDataLoader:
    """MNIST数据集加载器"""

    def __init__(self, data_dir='./data', batch_size=64, use_augmentation=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation

        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()

    def _get_train_transform(self):
        """获取训练数据预处理 transform"""
        if self.use_augmentation:
            return transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

    def _get_test_transform(self):
        """获取测试数据预处理 transform"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def get_dataloaders(self):
        """获取训练集和测试集的 DataLoader"""
        train_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )

        test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

        return train_loader, test_loader

    def get_train_size(self):
        """获取训练集大小"""
        return len(torchvision.datasets.MNIST(
            root=self.data_dir, train=True, download=False))

    def get_test_size(self):
        """获取测试集大小"""
        return len(torchvision.datasets.MNIST(
            root=self.data_dir, train=False, download=False))


class HandwritingDataset(Dataset):
    """自定义手写数字数据集，用于加载用户绘制的数据"""

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, os.path.basename(img_path)


class ImagePreprocessor:
    """图像预处理器，用于处理用户上传的手写图像"""

    def __init__(self, target_size=(28, 28)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def preprocess(self, image):
        """
        预处理图像

        参数:
            image: PIL Image对象或numpy数组

        返回:
            tensor: 预处理后的图像张量
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if not isinstance(image, Image.Image):
            raise ValueError("Unsupported image type")

        return self.transform(image)

    def preprocess_multiple(self, images):
        """批量预处理图像"""
        return torch.stack([self.preprocess(img) for img in images])


if __name__ == "__main__":
    loader = MNISTDataLoader(batch_size=64, use_augmentation=True)
    train_loader, test_loader = loader.get_dataloaders()

    print(f"训练集大小: {loader.get_train_size()}")
    print(f"测试集大小: {loader.get_test_size()}")

    for images, labels in train_loader:
        print(f"批次形状: {images.shape}, 标签形状: {labels.shape}")
        break
