"""
训练和评估模块
负责模型训练、验证、测试以及训练过程可视化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from pathlib import Path


class Trainer:
    """模型训练器"""

    def __init__(self, model, train_loader, test_loader, device='cpu', save_dir='models/checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []

    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def evaluate(self):
        """评估模型"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss = test_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        return test_loss, test_acc, all_preds, all_labels

    def train(self, epochs, save_best=True):
        """完整训练流程"""
        best_acc = 0.0
        best_model_path = None

        print(f"开始训练，设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            test_loss, test_acc, _, _ = self.evaluate()

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accs.append(test_acc)

            self.scheduler.step()

            print(f'\nEpoch {epoch}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print(f'  Overfitting Gap: {train_acc - test_acc:.2f}%')

            if save_best and test_acc > best_acc:
                best_acc = test_acc
                best_model_path = self.save_dir / f'lenet5_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'accuracy': test_acc,
                    'loss': test_loss,
                }, best_model_path)
                print(f'  [NEW BEST] 模型已保存: {best_model_path}')

            checkpoint_path = self.save_dir / f'lenet5_epoch{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'accuracy': test_acc,
                'loss': test_loss,
            }, checkpoint_path)

            print()

        return best_model_path, best_acc

    def get_metrics(self):
        """获取训练指标"""
        return {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'test_losses': self.test_losses,
            'test_accs': self.test_accs
        }


class Evaluator:
    """模型评估器"""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def predict(self, images):
        """单次或批量预测"""
        if not isinstance(images, torch.Tensor):
            raise ValueError("输入必须是 torch.Tensor")

        if images.dim() == 3:
            images = images.unsqueeze(0)

        images = images.to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = probabilities.max(1)

        return predicted.cpu().numpy(), confidences.cpu().numpy(), probabilities.cpu().numpy()

    def evaluate(self, test_loader):
        """在测试集上评估"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100. * correct / total
        return accuracy, all_preds, all_labels

    def measure_inference_time(self, image, num_runs=100):
        """测量推理时间"""
        if not isinstance(image, torch.Tensor):
            raise ValueError("输入必须是 torch.Tensor")

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():
            _ = self.model(image)

        times = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = self.model(image)
            times.append((time.time() - start) * 1000)

        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }


def plot_training_history(metrics, save_path=None):
    """绘制训练历史曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(metrics['train_losses']) + 1)

    ax1.plot(epochs, metrics['train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, metrics['test_losses'], 'r-', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, metrics['train_accs'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, metrics['test_accs'], 'r-', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"训练曲线已保存: {save_path}")

    return fig


if __name__ == "__main__":
    from models.lenet5 import LeNet5
    from data.data_loader import MNISTDataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = LeNet5()
    data_loader = MNISTDataLoader(batch_size=128)
    train_loader, test_loader = data_loader.get_dataloaders()

    trainer = Trainer(model, train_loader, test_loader, device)
    trainer.train(epochs=2)
