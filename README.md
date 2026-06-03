# LeNet-5 手写数字识别系统

基于卷积神经网络的手写数字识别系统，使用PyTorch实现LeNet-5模型，支持MNIST数据集训练和pygame可视化交互。

## 快速搭建

```bash
# 1. 创建虚拟环境
# python
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
# conda
conda create -n pytorch-lenet5 python=3.10
conda activate pytorch-lenet5


# 2. 安装依赖
pip install -r requirements.txt

# 3. pytorch
# cpu版本
# pip install torch torchvision torchaudio
# GPU版本（cuda13.0）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 4. 运行程序
python main.py train 20      # 训练模型
python main.py eval          # 评估模型
python main.py ui            # 启动手写界面
python main.py visualize     # 可视化特征图
```

## 依赖列表

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
pillow>=9.5.0
pygame>=2.5.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

## 项目结构

```
MNIST/
├── models/                     # 模型模块
│   ├── lenet5.py             # LeNet-5模型定义
│   └── model_manager.py       # 模型管理
├── data/                      # 数据模块
│   └── data_loader.py         # 数据加载
├── train/                     # 训练模块
│   └── trainer.py             # 训练器和评估器
├── ui/                       # 界面模块
│   └── handwriting_ui.py      # pygame界面
├── visualization/            # 可视化模块
│   └── visualizer.py         # 特征图可视化
├── outputs/                  # 输出目录
├── doc/                      # 文档目录
├── main.py                   # 主程序入口
└── README.md
```

## 功能特性

- LeNet-5模型实现，支持BatchNorm和Dropout
- MNIST数据集自动下载和预处理
- 模型训练、评估、性能测量
- pygame手写输入界面
- 特征图可视化
- 训练过程曲线可视化

## 使用方法

```bash
# 训练模型（默认10个epoch）
python main.py train 20

# 评估模型
python main.py eval

# 测量性能指标
python main.py perf

# 可视化特征图
python main.py visualize

# 启动手写识别界面
python main.py ui

# 完整演示（评估+界面）
python main.py demo

# 交互模式
python main.py
```

## 性能指标

| 指标 | 目标值 | 实际值 |
|------|--------|--------|
| 测试集准确率 | ≥99.0% | 98.73% |
| 测试集损失值 | ≤0.05 | 0.0399 |
| 平均推理时间 | ≤10ms | 0.02ms |
| 模型参数量 | ≤10万 | 61,750 |

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- Windows/Linux/Mac

## License

MIT License
