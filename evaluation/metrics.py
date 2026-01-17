"""
模型评估指标计算模块
包含 Accuracy, Precision, Recall, F1-Score 等指标的计算
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    """
    模型评估器
    计算分类任务的各项性能指标
    """
    
    def __init__(self, class_names=None):
        """
        Args:
            class_names: 类别名称列表，用于显示
        """
        if class_names is None:
            self.class_names = ['正常', '突增', '丢失', '无功']
        else:
            self.class_names = class_names
        
        self.num_classes = len(self.class_names)
    
    def calculate_metrics(self, y_true, y_pred):
        """
        计算所有评估指标
        
        Args:
            y_true: 真实标签 (numpy array or tensor)
            y_pred: 预测标签 (numpy array or tensor)
            
        Returns:
            metrics: 包含所有指标的字典
        """
        # 转换为 numpy array
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        # 确保是一维数组
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # 计算整体指标
        accuracy = accuracy_score(y_true, y_pred)
        
        # 计算每个类别的指标 (macro average)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # 计算加权平均指标 (weighted average)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 计算每个类别的详细指标
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def print_metrics(self, metrics, model_name="Model"):
        """
        打印评估指标
        
        Args:
            metrics: calculate_metrics 返回的指标字典
            model_name: 模型名称
        """
        print("\n" + "=" * 70)
        print(f"📊 {model_name} 评估结果")
        print("=" * 70)
        
        print(f"\n【整体性能指标】")
        print(f"  ✓ Accuracy (准确率):          {metrics['accuracy']*100:.2f}%")
        print(f"  ✓ Precision (精确率 - Macro):  {metrics['precision_macro']*100:.2f}%")
        print(f"  ✓ Recall (召回率 - Macro):     {metrics['recall_macro']*100:.2f}%")
        print(f"  ✓ F1-Score (Macro):            {metrics['f1_macro']*100:.2f}%")
        
        print(f"\n【加权平均指标】")
        print(f"  ✓ Precision (Weighted):        {metrics['precision_weighted']*100:.2f}%")
        print(f"  ✓ Recall (Weighted):           {metrics['recall_weighted']*100:.2f}%")
        print(f"  ✓ F1-Score (Weighted):         {metrics['f1_weighted']*100:.2f}%")
        
        print(f"\n【各类别详细指标】")
        print(f"{'类别':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 50)
        for i, class_name in enumerate(self.class_names):
            precision = metrics['precision_per_class'][i] * 100
            recall = metrics['recall_per_class'][i] * 100
            f1 = metrics['f1_per_class'][i] * 100
            print(f"{class_name:<10} {precision:>6.2f}%      {recall:>6.2f}%      {f1:>6.2f}%")
        
        print("=" * 70 + "\n")
    
    def plot_confusion_matrix(self, confusion_matrix, save_path=None, model_name="Model"):
        """
        绘制混淆矩阵
        
        Args:
            confusion_matrix: 混淆矩阵
            save_path: 保存路径
            model_name: 模型名称
        """
        plt.figure(figsize=(10, 8))
        
        # 绘制热力图
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': '样本数量'})
        
        plt.title(f'{model_name} - 混淆矩阵', fontsize=14, pad=20)
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 混淆矩阵已保存到: {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, metrics_dict, save_path=None):
        """
        绘制多个模型的指标对比图
        
        Args:
            metrics_dict: {model_name: metrics} 字典
            save_path: 保存路径
        """
        model_names = list(metrics_dict.keys())
        num_models = len(model_names)
        
        # 准备数据
        accuracies = [metrics_dict[m]['accuracy'] * 100 for m in model_names]
        precisions = [metrics_dict[m]['precision_macro'] * 100 for m in model_names]
        recalls = [metrics_dict[m]['recall_macro'] * 100 for m in model_names]
        f1_scores = [metrics_dict[m]['f1_macro'] * 100 for m in model_names]
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 颜色
        colors = plt.cm.Set3(range(num_models))
        
        # Accuracy
        axes[0, 0].bar(model_names, accuracies, color=colors, alpha=0.8)
        axes[0, 0].set_title('Accuracy (准确率)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('百分比 (%)', fontsize=10)
        axes[0, 0].set_ylim([0, 100])
        axes[0, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 2, f'{v:.2f}%', ha='center', fontsize=9)
        
        # Precision
        axes[0, 1].bar(model_names, precisions, color=colors, alpha=0.8)
        axes[0, 1].set_title('Precision (精确率 - Macro)', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('百分比 (%)', fontsize=10)
        axes[0, 1].set_ylim([0, 100])
        axes[0, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(precisions):
            axes[0, 1].text(i, v + 2, f'{v:.2f}%', ha='center', fontsize=9)
        
        # Recall
        axes[1, 0].bar(model_names, recalls, color=colors, alpha=0.8)
        axes[1, 0].set_title('Recall (召回率 - Macro)', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('百分比 (%)', fontsize=10)
        axes[1, 0].set_ylim([0, 100])
        axes[1, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(recalls):
            axes[1, 0].text(i, v + 2, f'{v:.2f}%', ha='center', fontsize=9)
        
        # F1-Score
        axes[1, 1].bar(model_names, f1_scores, color=colors, alpha=0.8)
        axes[1, 1].set_title('F1-Score (Macro)', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('百分比 (%)', fontsize=10)
        axes[1, 1].set_ylim([0, 100])
        axes[1, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(f1_scores):
            axes[1, 1].text(i, v + 2, f'{v:.2f}%', ha='center', fontsize=9)
        
        plt.suptitle('模型性能指标对比', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 对比图已保存到: {save_path}")
        
        plt.show()
    
    def save_metrics_to_file(self, metrics, model_name, save_path):
        """
        将评估指标保存到文本文件
        
        Args:
            metrics: 评估指标字典
            model_name: 模型名称
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"{model_name} 评估结果\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("【整体性能指标】\n")
            f.write(f"  Accuracy (准确率):          {metrics['accuracy']*100:.2f}%\n")
            f.write(f"  Precision (精确率 - Macro):  {metrics['precision_macro']*100:.2f}%\n")
            f.write(f"  Recall (召回率 - Macro):     {metrics['recall_macro']*100:.2f}%\n")
            f.write(f"  F1-Score (Macro):            {metrics['f1_macro']*100:.2f}%\n\n")
            
            f.write("【加权平均指标】\n")
            f.write(f"  Precision (Weighted):        {metrics['precision_weighted']*100:.2f}%\n")
            f.write(f"  Recall (Weighted):           {metrics['recall_weighted']*100:.2f}%\n")
            f.write(f"  F1-Score (Weighted):         {metrics['f1_weighted']*100:.2f}%\n\n")
            
            f.write("【各类别详细指标】\n")
            f.write(f"{'类别':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
            f.write("-" * 50 + "\n")
            for i, class_name in enumerate(self.class_names):
                precision = metrics['precision_per_class'][i] * 100
                recall = metrics['recall_per_class'][i] * 100
                f1 = metrics['f1_per_class'][i] * 100
                f.write(f"{class_name:<10} {precision:>6.2f}%      {recall:>6.2f}%      {f1:>6.2f}%\n")
            
            f.write("\n【混淆矩阵】\n")
            f.write(str(metrics['confusion_matrix']) + "\n")
            f.write("=" * 70 + "\n")
        
        print(f"📄 评估报告已保存到: {save_path}")


def calculate_class_statistics(y_true, y_pred, num_classes=4):
    """
    计算每个类别的统计信息
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        num_classes: 类别数量
        
    Returns:
        stats: 统计信息字典
    """
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    stats = {}
    for c in range(num_classes):
        true_mask = (y_true == c)
        pred_mask = (y_pred == c)
        
        # True Positive, False Positive, False Negative, True Negative
        tp = np.sum(true_mask & pred_mask)
        fp = np.sum(~true_mask & pred_mask)
        fn = np.sum(true_mask & ~pred_mask)
        tn = np.sum(~true_mask & ~pred_mask)
        
        total = np.sum(true_mask)
        correct = np.sum(true_mask & pred_mask)
        
        stats[c] = {
            'total': int(total),
            'correct': int(correct),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'accuracy': correct / total if total > 0 else 0
        }
    
    return stats
