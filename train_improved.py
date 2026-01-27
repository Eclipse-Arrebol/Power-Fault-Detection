"""
改进版训练脚本 - 针对"丢失"类型分类优化

主要改进：
1. 可调整的类别权重配置
2. Focal Loss 参数实验
3. 后处理阈值调整
"""
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os

from src.dataset import PowerGridDataset
from src.models import CausalGCN_LSTM
from src.loss.causal_loss import create_causal_loss
from plot.causal_plot import plot_causal_graph, analyze_causal_structure

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 🔧 实验配置：改进"丢失"类型分类
# ============================================================
WEIGHT_CONFIGS = {
    'original': [2.0, 50.0, 50.0, 20.0],        # 原始权重
    'reduce_missing': [2.0, 50.0, 25.0, 20.0],  # 降低丢失权重（推荐）
    'balanced': [2.0, 30.0, 30.0, 20.0],        # 平衡权重
    'aggressive': [2.0, 50.0, 15.0, 20.0]       # 激进降低丢失权重
}

# 选择权重配置
WEIGHT_CONFIG = 'reduce_missing'  # 改这里选择不同配置

# Focal Loss 参数
FOCAL_GAMMA = 2.5  # 默认2.0，增加到2.5或3.0更关注难样本

# 其他配置
SEQ_LEN = 12
BATCH_SIZE = 32
NUM_EPOCHS = 100


def predict_with_threshold(logits, missing_threshold=0.7):
    """
    使用阈值调整预测，减少"丢失"类的误判
    
    Args:
        logits: [B, N, C] 模型输出
        missing_threshold: 丢失类的概率阈值，默认0.7
        
    Returns:
        pred: 调整后的预测结果
    """
    probs = torch.softmax(logits, dim=-1)  # [B, N, C]
    pred = logits.argmax(dim=-1)  # 原始预测
    
    # 对预测为"丢失"(class 2)但概率不够高的，重新分类
    class_2_mask = (pred == 2)
    class_2_prob = probs[:, :, 2]
    low_confidence = class_2_mask & (class_2_prob < missing_threshold)
    
    if low_confidence.any():
        # 屏蔽丢失类，选择次优类别
        probs_copy = probs.clone()
        probs_copy[:, :, 2] = -float('inf')
        alternative_pred = probs_copy.argmax(dim=-1)
        pred[low_confidence] = alternative_pred[low_confidence]
    
    return pred


def train_improved_causal_gcn_lstm():
    """
    改进版 CausalGCN_LSTM 训练
    """
    print("=" * 60)
    print("🧠 改进版 CausalGCN_LSTM 训练")
    print(f"   配置: {WEIGHT_CONFIG}")
    print(f"   权重: {WEIGHT_CONFIGS[WEIGHT_CONFIG]}")
    print(f"   Focal Gamma: {FOCAL_GAMMA}")
    print("=" * 60)
    
    # 1. 准备数据
    print("\n>>> [1/6] 加载数据集...")
    dataset = PowerGridDataset(dataset_path="dataset")
    
    X, Y, edge_index, edge_weight, node_mask = dataset.get_temporal_tensors(seq_len=SEQ_LEN)
    num_samples = X.shape[0]
    num_nodes = X.shape[2]
    num_features = X.shape[3]
    
    print(f">>> 数据形状: {X.shape}")
    print(f">>> 节点数: {num_nodes}, 特征数: {num_features}")
    
    # 统计各类别样本数
    print("\n>>> 数据集类别分布:")
    for c in range(4):
        count = (Y == c).sum().item()
        print(f"    Class {c}: {count} 样本")
    
    # 2. 初始化设备
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        print(f"\n>>> 检测到 CUDA: {torch.version.cuda}")
    else:
        print("\n>>> 使用 CPU 训练")
    
    edge_index_tensor = edge_index.to(device)
    edge_weight_tensor = edge_weight.to(device)
    node_mask = node_mask.to(device)
    
    # 3. 划分训练/测试集
    indices = list(range(num_samples))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, shuffle=False)
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    print(f"\n>>> 训练集: {len(train_idx)} 样本, 测试集: {len(test_idx)} 样本")
    
    # 4. 初始化模型
    print(f"\n>>> [2/6] 初始化模型...")
    edge_index_np = edge_index.numpy()
    
    model = CausalGCN_LSTM(
        num_nodes=num_nodes,
        num_features=num_features,
        num_classes=4,
        edge_index=edge_index_np,
        admittance_matrix=None,
        source_node=0,
        gcn_hidden=64,
        lstm_hidden=128,
        num_gcn_layers=2,
        num_lstm_layers=2,
        dropout=0.3
    ).to(device)
    
    # 5. 初始化损失函数（使用调整后的权重）
    print(f"\n>>> [3/6] 初始化损失函数...")
    class_weights = torch.tensor(WEIGHT_CONFIGS[WEIGHT_CONFIG]).to(device)
    print(f">>> 类别权重: {class_weights.cpu().tolist()}")
    
    node_depths = model.node_depths.clone()
    
    criterion = create_causal_loss(
        class_weights=class_weights,
        node_depths=node_depths,
        use_focal_loss=True,
        focal_gamma=FOCAL_GAMMA,
        lambda_root=0.5,
        lambda_sparse=0.01,
        lambda_physics=0.1
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # 6. 训练循环
    print(f"\n>>> [4/6] 开始训练...")
    loss_history = {'total': [], 'anomaly': [], 'sparse': []}
    class_metrics = {'precision': [], 'recall': [], 'f1': []}
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_anomaly_loss = 0
        total_sparse_loss = 0
        
        perm = torch.randperm(len(train_idx))
        for i in range(0, len(train_idx), BATCH_SIZE):
            batch_indices = perm[i:i + BATCH_SIZE]
            current_batch_size = len(batch_indices)
            
            x_batch = X_train[batch_indices].to(device, non_blocking=True)
            y_batch = Y_train[batch_indices].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(x_batch, return_causal=True)
            
            loss, loss_dict = criterion(
                model_outputs=outputs,
                anomaly_labels=y_batch,
                root_cause_labels=None,
                x=x_batch,
                model=model
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss_dict.get('total', loss.item())
            total_anomaly_loss += loss_dict.get('anomaly', 0)
            total_sparse_loss += loss_dict.get('sparse', 0)
        
        scheduler.step()
        
        num_batches = len(train_idx) // BATCH_SIZE + 1
        avg_loss = total_loss / num_batches
        avg_anomaly = total_anomaly_loss / num_batches
        avg_sparse = total_sparse_loss / num_batches
        
        loss_history['total'].append(avg_loss)
        loss_history['anomaly'].append(avg_anomaly)
        loss_history['sparse'].append(avg_sparse)
        
        # 每10个epoch评估一次"丢失"类的性能
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # 在验证集上快速评估
                val_size = min(len(test_idx), 500)
                x_val = X_test[:val_size].to(device)
                y_val = Y_test[:val_size].to(device)
                
                outputs = model(x_val)
                anomaly_logits = outputs['anomaly_logits']
                
                # 使用阈值调整的预测
                pred = predict_with_threshold(anomaly_logits, missing_threshold=0.7)
                
                # 计算"丢失"类的指标
                mask = node_mask.repeat(val_size).cpu()
                pred_flat = pred.view(-1).cpu()
                y_flat = y_val.view(-1).cpu()
                
                # 只统计"丢失"类 (class 2)
                class_2_mask = (y_flat == 2) & mask
                if class_2_mask.sum() > 0:
                    tp = ((pred_flat == 2) & (y_flat == 2) & mask).sum().item()
                    fp = ((pred_flat == 2) & (y_flat != 2) & mask).sum().item()
                    fn = ((pred_flat != 2) & (y_flat == 2) & mask).sum().item()
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    print(f"    Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                          f"丢失类 - P: {precision*100:.1f}% R: {recall*100:.1f}% F1: {f1*100:.1f}%")
                else:
                    print(f"    Epoch {epoch:03d} | Loss: {avg_loss:.4f}")
            
            model.train()
    
    # 7. 最终评估
    print("\n>>> [5/6] 评估模型...")
    model.eval()
    correct = 0
    total = 0
    class_correct = [0, 0, 0, 0]
    class_total = [0, 0, 0, 0]
    class_tp = [0, 0, 0, 0]
    class_fp = [0, 0, 0, 0]
    class_fn = [0, 0, 0, 0]
    
    test_batch_size = BATCH_SIZE * 2
    with torch.no_grad():
        for i in range(0, len(test_idx), test_batch_size):
            end_idx = min(i + test_batch_size, len(test_idx))
            batch_indices = range(i, end_idx)
            current_batch_size = len(batch_indices)
            
            x_batch = X_test[batch_indices].to(device)
            y_batch = Y_test[batch_indices].to(device)
            
            outputs = model(x_batch)
            anomaly_logits = outputs['anomaly_logits']
            
            # 🔥 使用阈值调整的预测
            pred = predict_with_threshold(anomaly_logits, missing_threshold=0.7)
            
            mask_expanded = node_mask.repeat(current_batch_size)
            pred_flat = pred.view(-1)
            y_flat = y_batch.view(-1)
            
            valid_mask = mask_expanded
            correct += (pred_flat[valid_mask] == y_flat[valid_mask]).sum().item()
            total += valid_mask.sum().item()
            
            for c in range(4):
                c_mask = valid_mask & (y_flat == c)
                class_total[c] += c_mask.sum().item()
                class_correct[c] += (pred_flat[c_mask] == y_flat[c_mask]).sum().item()
                
                # 计算 TP, FP, FN
                class_tp[c] += ((pred_flat == c) & (y_flat == c) & valid_mask).sum().item()
                class_fp[c] += ((pred_flat == c) & (y_flat != c) & valid_mask).sum().item()
                class_fn[c] += ((pred_flat != c) & (y_flat == c) & valid_mask).sum().item()
    
    acc = correct / total if total > 0 else 0
    
    print("=" * 60)
    print(f"✅ 测试集总准确率: {acc * 100:.2f}%")
    print("\n【各类别详细指标】")
    class_names = ['正常', '突增', '丢失', '无功']
    
    for c in range(4):
        accuracy = 100 * class_correct[c] / max(class_total[c], 1)
        precision = 100 * class_tp[c] / max(class_tp[c] + class_fp[c], 1)
        recall = 100 * class_tp[c] / max(class_tp[c] + class_fn[c], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1)
        
        marker = "🎯" if c == 2 else "  "
        print(f"{marker} {class_names[c]}: Acc={accuracy:.1f}% | P={precision:.1f}% | R={recall:.1f}% | F1={f1:.1f}%")
    
    print("=" * 60)
    
    # 保存模型
    save_dir = f"result/improved_{WEIGHT_CONFIG}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save(model.state_dict(), f"{save_dir}/model.pth")
    checkpoint = {
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'accuracy': acc,
        'config': {
            'weight_config': WEIGHT_CONFIG,
            'class_weights': WEIGHT_CONFIGS[WEIGHT_CONFIG],
            'focal_gamma': FOCAL_GAMMA,
            'missing_threshold': 0.7
        }
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")
    print(f"\n💾 模型已保存到: {save_dir}/")
    
    # 绘制因果图
    print("\n>>> [6/6] 绘制因果图...")
    causal_matrix = model.get_causal_matrix().detach().cpu().numpy()
    np.save(f"{save_dir}/causal_matrix.npy", causal_matrix)
    
    analyze_causal_structure(causal_matrix, top_k=10)
    plot_causal_graph(causal_matrix, threshold=0.1, save_path=f"{save_dir}/causal_graph.png")
    
    # 绘制训练曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(loss_history['total'], color='blue')
    axes[0].set_title(f"Total Loss ({WEIGHT_CONFIG})")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True)
    
    axes[1].plot(loss_history['anomaly'], color='green')
    axes[1].set_title("Anomaly Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True)
    
    axes[2].plot(loss_history['sparse'], color='orange')
    axes[2].set_title("Sparsity Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_loss.png", dpi=300)
    plt.show()
    
    return model, acc


if __name__ == "__main__":
    print("\n" + "🎯" * 30)
    print("改进版训练：针对'丢失'类型优化")
    print("🎯" * 30)
    
    train_improved_causal_gcn_lstm()
