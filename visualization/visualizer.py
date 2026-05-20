"""
可视化模块 - 特征图可视化和训练过程可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torchvision


class FeatureMapVisualizer:
    """特征图可视化工具"""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.model.eval()
        self.device = device

    def get_feature_maps(self, image_tensor):
        """获取输入图像的特征图"""
        if not isinstance(image_tensor, torch.Tensor):
            raise ValueError("输入必须是 torch.Tensor")

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            features = self.model.get_feature_maps(image_tensor)

        return features

    def visualize_feature_maps(self, image_tensor, save_path=None, max_channels=6):
        """可视化特征图"""
        features = self.get_feature_maps(image_tensor)

        layer_names = ['Conv1 (6 channels)', 'Conv2 (16 channels)', 'Conv3 (120 channels)']

        fig = plt.figure(figsize=(15, 6))
        gs = gridspec.GridSpec(3, 6, figure=fig)

        for layer_idx, (feature, name) in enumerate(zip(features, layer_names)):
            feature = feature.squeeze(0)
            num_channels = min(feature.size(0), max_channels)

            for ch in range(num_channels):
                row = layer_idx
                col = ch
                ax = fig.add_subplot(gs[row, col])
                feat_map = feature[ch].cpu().numpy()
                ax.imshow(feat_map, cmap='viridis')
                ax.axis('off')
                if ch == 0:
                    ax.set_title(name, fontsize=8, loc='left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"特征图已保存: {save_path}")

        return fig

    def visualize_conv_filters(self, layer_idx=0, save_path=None):
        """可视化卷积层滤波器"""
        if layer_idx == 0:
            filters = self.model.conv1.weight.data
        elif layer_idx == 1:
            filters = self.model.conv2.weight.data
        elif layer_idx == 2:
            filters = self.model.conv3.weight.data
        else:
            raise ValueError("layer_idx 必须是 0, 1, 或 2")

        num_filters = min(filters.size(0), 16)
        fig, axes = plt.subplots(2, 8, figsize=(14, 4))
        for i in range(num_filters):
            row, col = i // 8, i % 8
            filter_img = filters[i].squeeze(0).cpu().numpy()
            if filter_img.ndim == 2:
                axes[row, col].imshow(filter_img, cmap='gray')
            else:
                axes[row, col].imshow(filter_img[0], cmap='gray')
            axes[row, col].axis('off')

        plt.suptitle(f'Conv{layer_idx + 1} Filters', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"滤波器图已保存: {save_path}")

        return fig


class TrainingVisualizer:
    """训练过程可视化工具"""

    @staticmethod
    def plot_training_curves(metrics, save_path=None):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(metrics['train_losses']) + 1)

        ax1.plot(epochs, metrics['train_losses'], 'b-', linewidth=2, label='Training Loss')
        ax1.plot(epochs, metrics['test_losses'], 'r-', linewidth=2, label='Test Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Test Loss', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, metrics['train_accs'], 'b-', linewidth=2, label='Training Accuracy')
        ax2.plot(epochs, metrics['test_accs'], 'r-', linewidth=2, label='Test Accuracy')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Test Accuracy', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练曲线已保存: {save_path}")

        return fig

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path=None):
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=range(10), yticklabels=range(10))
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"混淆矩阵已保存: {save_path}")

        return fig


class ImageVisualizer:
    """图像可视化工具"""

    @staticmethod
    def show_sample_images(dataset, num_samples=10, save_path=None):
        """显示样本图像"""
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 2))

        for i in range(num_samples):
            img, label = dataset[i]
            if isinstance(img, torch.Tensor):
                img = img.squeeze().numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"样本图像已保存: {save_path}")

        return fig

    @staticmethod
    def visualize_predictions(images, true_labels, pred_labels, confidences, num_samples=10, save_path=None):
        """可视化预测结果"""
        num_samples = min(num_samples, len(images))
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

        for i in range(num_samples):
            img = images[i]
            if isinstance(img, torch.Tensor):
                img = img.squeeze().cpu().numpy()
            elif img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)

            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'True: {true_labels[i]}')
            axes[0, i].axis('off')

            axes[1, i].text(0.5, 0.5, f'Pred: {pred_labels[i]}\nConf: {confidences[i]:.2f}',
                          ha='center', va='center', fontsize=10,
                          transform=axes[1, i].transAxes)
            color = 'green' if true_labels[i] == pred_labels[i] else 'red'
            axes[1, i].set_title(f'Pred: {pred_labels[i]}', color=color)
            axes[1, i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"预测结果可视化已保存: {save_path}")

        return fig


if __name__ == "__main__":
    print("可视化模块测试")
    print("请通过 main.py 运行主程序来使用可视化功能")
