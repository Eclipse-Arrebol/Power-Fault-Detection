"""
评估所有训练好的模型
加载模型检查点，在测试集上计算各项指标
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.model_selection import train_test_split

from src.dataset import PowerGridDataset
from src.models import GCN, TGCN, NeuralGrangerCausality, CausalGCN_LSTM
from evaluation.metrics import ModelEvaluator, calculate_class_statistics


def evaluate_gcn(dataset, evaluator, device):
    """
    评估 GCN 模型
    """
    print("\n" + "="*70)
    print("🔍 评估 GCN 模型")
    print("="*70)
    
    # 检查模型文件是否存在
    model_path = "result/gcn/checkpoint.pth"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    # 加载数据
    data_list = dataset.get_pyg_data_list()
    train_data, test_data = train_test_split(data_list, test_size=0.2, shuffle=False)
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model = GCN(num_features=6, num_classes=4).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 模型加载成功 (训练轮次: {checkpoint['epoch']})")
    
    # 收集预测结果
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for data in test_data:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            
            # 只统计有效节点
            mask = data.train_mask
            y_true_list.append(data.y[mask].cpu())
            y_pred_list.append(pred[mask].cpu())
    
    y_true = torch.cat(y_true_list).numpy()
    y_pred = torch.cat(y_pred_list).numpy()
    
    # 计算指标
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    evaluator.print_metrics(metrics, model_name="GCN")
    
    # 保存结果
    save_dir = "result/gcn"
    evaluator.save_metrics_to_file(metrics, "GCN", f"{save_dir}/evaluation_report.txt")
    evaluator.plot_confusion_matrix(metrics['confusion_matrix'], 
                                     save_path=f"{save_dir}/confusion_matrix.png",
                                     model_name="GCN")
    
    return metrics


def evaluate_tgcn(dataset, evaluator, device, seq_len=12, batch_size=64):
    """
    评估 TGCN 模型
    """
    print("\n" + "="*70)
    print("🔍 评估 TGCN 模型")
    print("="*70)
    
    # 检查模型文件是否存在
    model_path = "result/tgcn/checkpoint.pth"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    # 加载数据
    X, Y, edge_index, edge_weight, node_mask = dataset.get_temporal_tensors(seq_len=seq_len)
    num_samples = X.shape[0]
    num_features = X.shape[3]
    
    indices = list(range(num_samples))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, shuffle=False)
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    # 移动数据到设备
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    node_mask = node_mask.to(device)
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model = TGCN(num_features=num_features, num_classes=4, hidden_dim=64, num_layers=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 模型加载成功 (训练轮次: {checkpoint['epoch']})")
    
    # 收集预测结果
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for i in range(0, len(test_idx), batch_size):
            end_idx = min(i + batch_size, len(test_idx))
            batch_indices = range(i, end_idx)
            current_batch_size = len(batch_indices)
            
            x_batch = X_test[batch_indices].to(device)
            y_batch = Y_test[batch_indices].to(device)
            
            out = model(x_batch, edge_index, edge_weight)
            pred = out.argmax(dim=2)
            
            mask_expanded = node_mask.repeat(current_batch_size)
            pred_flat = pred.view(-1)
            y_flat = y_batch.view(-1)
            
            y_true_list.append(y_flat[mask_expanded].cpu())
            y_pred_list.append(pred_flat[mask_expanded].cpu())
    
    y_true = torch.cat(y_true_list).numpy()
    y_pred = torch.cat(y_pred_list).numpy()
    
    # 计算指标
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    evaluator.print_metrics(metrics, model_name="TGCN")
    
    # 保存结果
    save_dir = "result/tgcn"
    evaluator.save_metrics_to_file(metrics, "TGCN", f"{save_dir}/evaluation_report.txt")
    evaluator.plot_confusion_matrix(metrics['confusion_matrix'], 
                                     save_path=f"{save_dir}/confusion_matrix.png",
                                     model_name="TGCN")
    
    return metrics


def evaluate_ngc(dataset, evaluator, device, seq_len=12, batch_size=64):
    """
    评估 NGC 模型
    """
    print("\n" + "="*70)
    print("🔍 评估 NGC 模型")
    print("="*70)
    
    # 检查模型文件是否存在
    model_path = "result/ngc/checkpoint.pth"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    # 加载数据
    X, Y, edge_index, edge_weight, node_mask = dataset.get_temporal_tensors(seq_len=seq_len)
    num_samples = X.shape[0]
    num_nodes = X.shape[2]
    num_features = X.shape[3]
    
    indices = list(range(num_samples))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, shuffle=False)
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    # 移动数据到设备
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    node_mask = node_mask.to(device)
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model = NeuralGrangerCausality(
        num_nodes=num_nodes,
        num_features=num_features,
        num_classes=4,
        hidden_dim=64,
        num_layers=2,
        sparsity_lambda=checkpoint.get('sparsity_lambda', 0.01)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 模型加载成功 (训练轮次: {checkpoint['epoch']})")
    
    # 收集预测结果
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for i in range(0, len(test_idx), batch_size):
            end_idx = min(i + batch_size, len(test_idx))
            batch_indices = range(i, end_idx)
            current_batch_size = len(batch_indices)
            
            x_batch = X_test[batch_indices].to(device)
            y_batch = Y_test[batch_indices].to(device)
            
            out = model(x_batch, edge_index, edge_weight)
            pred = out.argmax(dim=2)
            
            mask_expanded = node_mask.repeat(current_batch_size)
            pred_flat = pred.view(-1)
            y_flat = y_batch.view(-1)
            
            y_true_list.append(y_flat[mask_expanded].cpu())
            y_pred_list.append(pred_flat[mask_expanded].cpu())
    
    y_true = torch.cat(y_true_list).numpy()
    y_pred = torch.cat(y_pred_list).numpy()
    
    # 计算指标
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    evaluator.print_metrics(metrics, model_name="NGC")
    
    # 保存结果
    save_dir = "result/ngc"
    evaluator.save_metrics_to_file(metrics, "NGC", f"{save_dir}/evaluation_report.txt")
    evaluator.plot_confusion_matrix(metrics['confusion_matrix'], 
                                     save_path=f"{save_dir}/confusion_matrix.png",
                                     model_name="NGC")
    
    return metrics


def evaluate_causal_gcn_lstm(dataset, evaluator, device, seq_len=12, batch_size=64):
    """
    评估 CausalGCN_LSTM 模型
    """
    print("\n" + "="*70)
    print("🔍 评估 CausalGCN_LSTM 模型")
    print("="*70)
    
    # 检查模型文件是否存在
    model_path = "result/causal_gcn_lstm/checkpoint.pth"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    # 加载数据
    X, Y, edge_index, edge_weight, node_mask = dataset.get_temporal_tensors(seq_len=seq_len)
    num_samples = X.shape[0]
    num_nodes = X.shape[2]
    num_features = X.shape[3]
    
    indices = list(range(num_samples))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, shuffle=False)
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    # 移动数据到设备
    edge_index_tensor = edge_index.to(device)
    edge_weight_tensor = edge_weight.to(device)
    node_mask = node_mask.to(device)
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    edge_index_np = edge_index.numpy()
    
    model = CausalGCN_LSTM(
        num_nodes=num_nodes,
        num_features=num_features,
        num_classes=4,
        edge_index=edge_index_np,
        admittance_matrix=None,
        source_node=0,
        gcn_hidden=checkpoint['config']['gcn_hidden'],
        lstm_hidden=checkpoint['config']['lstm_hidden'],
        num_gcn_layers=checkpoint['config']['num_gcn_layers'],
        num_lstm_layers=checkpoint['config']['num_lstm_layers'],
        dropout=0.3
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 模型加载成功 (训练轮次: {checkpoint['epoch']})")
    
    # 收集预测结果
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for i in range(0, len(test_idx), batch_size):
            end_idx = min(i + batch_size, len(test_idx))
            batch_indices = range(i, end_idx)
            current_batch_size = len(batch_indices)
            
            x_batch = X_test[batch_indices].to(device)
            y_batch = Y_test[batch_indices].to(device)
            
            outputs = model(x_batch)
            anomaly_logits = outputs['anomaly_logits']
            pred = anomaly_logits.argmax(dim=2)
            
            mask_expanded = node_mask.repeat(current_batch_size)
            pred_flat = pred.view(-1)
            y_flat = y_batch.view(-1)
            
            y_true_list.append(y_flat[mask_expanded].cpu())
            y_pred_list.append(pred_flat[mask_expanded].cpu())
    
    y_true = torch.cat(y_true_list).numpy()
    y_pred = torch.cat(y_pred_list).numpy()
    
    # 计算指标
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    evaluator.print_metrics(metrics, model_name="CausalGCN_LSTM")
    
    # 保存结果
    save_dir = "result/causal_gcn_lstm"
    evaluator.save_metrics_to_file(metrics, "CausalGCN_LSTM", f"{save_dir}/evaluation_report.txt")
    evaluator.plot_confusion_matrix(metrics['confusion_matrix'], 
                                     save_path=f"{save_dir}/confusion_matrix.png",
                                     model_name="CausalGCN_LSTM")
    
    return metrics


def main():
    """
    主函数：评估所有模型
    """
    print("\n" + "🎯" * 30)
    print("开始评估所有模型")
    print("🎯" * 30 + "\n")
    
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("\n加载数据集...")
    dataset = PowerGridDataset(dataset_path="dataset")
    
    # 初始化评估器
    evaluator = ModelEvaluator(class_names=['正常', '突增', '丢失', '无功'])
    
    # 评估各个模型
    all_metrics = {}
    
    # 1. 评估 GCN
    metrics = evaluate_gcn(dataset, evaluator, device)
    if metrics:
        all_metrics['GCN'] = metrics
    
    # 2. 评估 TGCN
    metrics = evaluate_tgcn(dataset, evaluator, device)
    if metrics:
        all_metrics['TGCN'] = metrics
    
    # 3. 评估 NGC
    metrics = evaluate_ngc(dataset, evaluator, device)
    if metrics:
        all_metrics['NGC'] = metrics
    
    # 4. 评估 CausalGCN_LSTM
    metrics = evaluate_causal_gcn_lstm(dataset, evaluator, device)
    if metrics:
        all_metrics['CausalGCN_LSTM'] = metrics
    
    # 绘制对比图
    if len(all_metrics) > 0:
        print("\n" + "="*70)
        print("📊 生成模型对比图...")
        print("="*70)
        evaluator.plot_metrics_comparison(all_metrics, save_path="result/model_comparison.png")
        
        # 保存对比报告
        save_comparison_report(all_metrics, "result/comparison_report.txt")
    
    print("\n" + "✅" * 30)
    print("所有模型评估完成！")
    print("✅" * 30 + "\n")


def save_comparison_report(metrics_dict, save_path):
    """
    保存模型对比报告
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("模型性能对比报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 表头
        f.write(f"{'模型':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 80 + "\n")
        
        # 每个模型的数据
        for model_name, metrics in metrics_dict.items():
            acc = metrics['accuracy'] * 100
            prec = metrics['precision_macro'] * 100
            rec = metrics['recall_macro'] * 100
            f1 = metrics['f1_macro'] * 100
            
            f.write(f"{model_name:<20} {acc:>6.2f}%      {prec:>6.2f}%      {rec:>6.2f}%      {f1:>6.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        
        # 找出最佳模型
        best_acc_model = max(metrics_dict.items(), key=lambda x: x[1]['accuracy'])
        best_f1_model = max(metrics_dict.items(), key=lambda x: x[1]['f1_macro'])
        
        f.write("\n【最佳模型】\n")
        f.write(f"  最高准确率: {best_acc_model[0]} ({best_acc_model[1]['accuracy']*100:.2f}%)\n")
        f.write(f"  最高F1分数: {best_f1_model[0]} ({best_f1_model[1]['f1_macro']*100:.2f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"📄 对比报告已保存到: {save_path}")


if __name__ == "__main__":
    main()
