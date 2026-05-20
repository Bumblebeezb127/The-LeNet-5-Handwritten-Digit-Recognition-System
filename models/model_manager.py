"""
模型管理模块 - 模型保存、加载、版本控制
"""

import torch
import os
from pathlib import Path


class ModelManager:
    """模型管理器，负责模型的保存和加载"""

    def __init__(self, model_dir="models/checkpoints"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, optimizer, epoch, accuracy, loss, filename=None):
        """保存模型及相关训练状态"""
        if filename is None:
            filename = f"lenet5_epoch{epoch}_acc{accuracy:.4f}.pth"

        filepath = self.model_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
            'loss': loss,
        }, filepath)
        print(f"模型已保存: {filepath}")
        return str(filepath)

    def load_model(self, model, filepath, device='cpu'):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        accuracy = checkpoint.get('accuracy', 0.0)
        loss = checkpoint.get('loss', 0.0)
        print(f"模型已加载: {filepath}")
        print(f"  - Epoch: {epoch}")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Loss: {loss:.4f}")
        return model, epoch, accuracy, loss

    def load_latest_model(self, model, device='cpu'):
        """加载最新的模型"""
        checkpoints = list(self.model_dir.glob("lenet5_*.pth"))
        if not checkpoints:
            print("未找到已保存的模型")
            return None

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return self.load_model(model, latest, device)

    def list_models(self):
        """列出所有保存的模型"""
        checkpoints = list(self.model_dir.glob("*.pth"))
        for cp in sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True):
            print(f"  {cp.name}")
        return checkpoints


if __name__ == "__main__":
    from models.lenet5 import LeNet5

    manager = ModelManager()
    model = LeNet5()

    print("\n可用的模型检查点:")
    manager.list_models()
