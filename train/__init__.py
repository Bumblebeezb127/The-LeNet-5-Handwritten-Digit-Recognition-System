"""
train package - 训练和评估模块
"""

from .trainer import Trainer, Evaluator, plot_training_history

__all__ = ['Trainer', 'Evaluator', 'plot_training_history']
