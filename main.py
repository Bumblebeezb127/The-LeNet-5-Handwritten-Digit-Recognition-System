"""
LeNet-5 手写数字识别系统 - 主程序
整合模型训练、评估、特征可视化以及pygame手写输入界面
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.lenet5 import LeNet5, count_parameters
from models.model_manager import ModelManager
from data.data_loader import MNISTDataLoader
from train.trainer import Trainer, Evaluator, plot_training_history
from visualization.visualizer import FeatureMapVisualizer, TrainingVisualizer, ImageVisualizer
from ui.handwriting_ui import HandwritingUI


class MNISTRecognizer:
    """MNIST手写数字识别系统主类"""

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.model_manager = ModelManager()
        self.data_loader = None
        self.trainer = None
        self.evaluator = None

        print("=" * 60)
        print("LeNet-5 手写数字识别系统")
        print("=" * 60)
        print(f"使用设备: {self.device}")

    def build_model(self, load_if_exists=True):
        """构建模型"""
        print("\n[1] 构建LeNet-5模型...")

        self.model = LeNet5().to(self.device)
        param_count = count_parameters(self.model)
        print(f"    模型参数量: {param_count:,}")

        if load_if_exists:
            try:
                self.model, epoch, accuracy, loss = self.model_manager.load_latest_model(
                    self.model, self.device)
                print(f"    已加载已训练模型")
                return True
            except:
                print("    未找到已保存的模型，将从头训练")
                return False
        return False

    def load_data(self, batch_size=128):
        """加载数据"""
        print("\n[2] 加载MNIST数据集...")
        self.data_loader = MNISTDataLoader(batch_size=batch_size, use_augmentation=True)
        train_loader, test_loader = self.data_loader.get_dataloaders()
        print(f"    训练集: {self.data_loader.get_train_size()} 样本")
        print(f"    测试集: {self.data_loader.get_test_size()} 样本")
        return train_loader, test_loader

    def train_model(self, epochs=10):
        """训练模型"""
        if self.model is None:
            self.build_model(load_if_exists=False)

        train_loader, test_loader = self.load_data()

        print(f"\n[3] 训练模型 (共 {epochs} 个epoch)...")

        self.trainer = Trainer(
            self.model, train_loader, test_loader,
            device=self.device
        )

        best_path, best_acc = self.trainer.train(epochs=epochs)
        metrics = self.trainer.get_metrics()

        print(f"\n    训练完成!")
        print(f"    最佳测试准确率: {best_acc:.2f}%")

        return metrics

    def evaluate_model(self):
        """评估模型"""
        if self.model is None:
            print("错误: 模型未加载，请先训练或加载模型")
            return None

        _, test_loader = self.load_data()

        print("\n[4] 评估模型...")

        self.evaluator = Evaluator(self.model, self.device)
        accuracy, preds, labels = self.evaluator.evaluate(test_loader)

        print(f"    测试集准确率: {accuracy:.2f}%")

        from sklearn.metrics import classification_report
        print("\n    分类报告:")
        print(classification_report(labels, preds, digits=4))

        sample_images = []
        sample_labels = []
        sample_preds = []

        for i, (images, targets) in enumerate(test_loader):
            if i == 0:
                for j in range(min(10, len(images))):
                    sample_images.append(images[j])
                    sample_labels.append(targets[j].item())
                    pred, _, _ = self.evaluator.predict(images[j])
                    sample_preds.append(pred[0])
                break

        ImageVisualizer.visualize_predictions(
            sample_images, sample_labels, sample_preds,
            [1.0] * len(sample_labels),
            num_samples=len(sample_images),
            save_path='outputs/prediction_samples.png'
        )

        return accuracy

    def measure_performance(self):
        """测量模型性能指标"""
        if self.model is None:
            print("错误: 模型未加载")
            return

        print("\n[5] 性能指标测量...")

        _, test_loader = self.load_data()

        self.evaluator = Evaluator(self.model, self.device)

        test_loss = 0.0
        correct = 0
        total = 0
        criterion = torch.nn.CrossEntropyLoss()

        import time
        times = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                start = time.time()
                outputs = self.model(images)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = (time.time() - start) * 1000
                times.extend([elapsed / images.size(0)] * images.size(0))

                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / total
        avg_time = sum(times) / len(times)

        print(f"    测试集损失值: {test_loss:.4f}")
        print(f"    测试集准确率: {accuracy:.2f}%")
        print(f"    平均推理时间: {avg_time:.2f}ms/样本")
        print(f"    模型参数量: {count_parameters(self.model):,}")

        return {
            'loss': test_loss,
            'accuracy': accuracy,
            'inference_time': avg_time
        }

    def visualize_features(self, image_index=0):
        """可视化特征图"""
        if self.model is None:
            print("错误: 模型未加载")
            return

        print("\n[6] 特征图可视化...")

        _, test_loader = self.load_data()

        for i, (images, _) in enumerate(test_loader):
            if i == 0:
                image = images[image_index].to(self.device)
                break

        visualizer = FeatureMapVisualizer(self.model, self.device)
        visualizer.visualize_feature_maps(image, save_path='outputs/feature_maps.png')
        print("    特征图已保存: outputs/feature_maps.png")

        print("\n    可视化卷积滤波器...")
        for layer_idx in range(3):
            visualizer.visualize_conv_filters(layer_idx, save_path=f'outputs/conv{layer_idx+1}_filters.png')
            print(f"    Conv{layer_idx+1}滤波器已保存: outputs/conv{layer_idx+1}_filters.png")

    def run_interface(self):
        """运行pygame手写输入界面"""
        if self.model is None:
            self.build_model(load_if_exists=True)
            if self.model is None:
                print("错误: 无法加载模型，请先训练模型")
                return

        print("\n[7] 启动手写识别界面...")
        print("    关闭界面窗口以返回主菜单")

        ui = HandwritingUI(self.model, self.device)
        ui.run()

    def interactive_menu(self):
        """交互式菜单"""
        os.makedirs('outputs', exist_ok=True)

        while True:
            print("\n" + "=" * 40)
            print("请选择操作:")
            print("=" * 40)
            print("1. 训练模型")
            print("2. 评估模型")
            print("3. 测量性能指标")
            print("4. 可视化特征图")
            print("5. 启动手写识别界面")
            print("6. 显示所有选项")
            print("0. 退出")
            print("=" * 40)

            choice = input("\n请输入选项: ").strip()

            if choice == '1':
                epochs = int(input("请输入训练epoch数 (默认10): ") or "10")
                self.build_model(load_if_exists=False)
                self.train_model(epochs=epochs)

            elif choice == '2':
                self.build_model()
                self.evaluate_model()

            elif choice == '3':
                self.build_model()
                self.measure_performance()

            elif choice == '4':
                self.build_model()
                self.visualize_features()

            elif choice == '5':
                self.run_interface()

            elif choice == '6':
                self.build_model()
                self.measure_performance()
                self.evaluate_model()
                self.visualize_features()

            elif choice == '0':
                print("\n感谢使用LeNet-5手写数字识别系统!")
                break

            else:
                print("无效选项，请重新输入")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        recognizer = MNISTRecognizer()

        if command == 'train':
            epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            recognizer.build_model(load_if_exists=False)
            recognizer.train_model(epochs=epochs)

        elif command == 'eval':
            recognizer.build_model()
            recognizer.evaluate_model()

        elif command == 'perf':
            recognizer.build_model()
            recognizer.measure_performance()

        elif command == 'visualize':
            recognizer.build_model()
            recognizer.visualize_features()

        elif command == 'ui':
            recognizer.run_interface()

        elif command == 'demo':
            recognizer.build_model(load_if_exists=True)
            recognizer.measure_performance()
            recognizer.run_interface()

        else:
            print(f"未知命令: {command}")
            print("可用命令: train, eval, perf, visualize, ui, demo")

    else:
        recognizer = MNISTRecognizer()
        recognizer.interactive_menu()


if __name__ == "__main__":
    main()
