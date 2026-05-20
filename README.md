# LeNet-5 手写数字识别系统

## 项目概述

本项目使用经典的LeNet-5卷积神经网络模型，对MNIST手写数字数据集进行识别分类。

## 项目结构

```
MNIST/
├── models/                 # 模型定义模块
│   ├── lenet5.py          # LeNet-5模型架构
│   ├── model_manager.py   # 模型管理工具
│   └── __init__.py
├── data/                   # 数据加载模块
│   ├── data_loader.py     # 数据加载和预处理
│   └── __init__.py
├── train/                  # 训练模块
│   ├── trainer.py         # 训练器和评估器
│   └── __init__.py
├── ui/                     # 用户界面模块
│   ├── handwriting_ui.py  # pygame手写界面
│   └── __init__.py
├── visualization/         # 可视化模块
│   ├── visualizer.py      # 特征图可视化
│   └── __init__.py
├── outputs/                # 输出目录
├── doc/                    # 文档目录
├── main.py                # 主程序
└── README.md
```

## 功能特性

1. LeNet-5模型实现，支持BatchNorm和Dropout
2. MNIST数据集自动下载和预处理
3. 模型训练、评估、性能测量
4. pygame手写输入界面
5. 特征图可视化
6. 训练过程曲线可视化

## 使用方法

```bash
# 训练模型
python main.py train 20

# 评估模型
python main.py eval

# 测量性能
python main.py perf

# 可视化特征图
python main.py visualize

# 启动手写界面
python main.py ui

# 完整演示
python main.py demo
```

## 性能指标

- 测试集准确率: ≥99.0%
- 测试集损失值: ≤0.05
- 推理时间: ≤10ms/样本
- 模型参数量: ≤10万
