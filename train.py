import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os

# 导入我们在 src 里写的模块
from src.dataset import PowerGridDataset
from src.models import GCN, TGCN, TemporalGCN, NeuralGrangerCausality, CausalGCN_LSTM, create_causal_model
from src.loss.causal_loss import CausalLoss, create_causal_loss

# 导入画图函数
from plot.causal_plot import plot_causal_graph, analyze_causal_structure

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 配置: 选择模型类型
# ============================================================
USE_TEMPORAL = True  # True: 使用时序GNN (TGCN), False: 使用普通GCN
USE_NGC = False        # True: 使用神经格兰杰因果模型 (优先级最高)
USE_CAUSAL_LSTM = False  # True: 使用因果GCN-LSTM模型 (最高优先级)
SEQ_LEN = 12         # 时间窗口长度 (仅 TGCN 使用)
BATCH_SIZE = 32      # 批大小
SPARSITY_LAMBDA = 0.01  # 稀疏性正则化系数 (NGC 使用)


def train_causal_gcn_lstm():
    """
    使用 CausalGCN_LSTM 因果感知模型训练
    
    特点:
    1. 物理引导的因果注意力机制
    2. 使用因果损失函数 (分类损失 + 根因损失 + 稀疏性损失 + 物理一致性损失)
    3. 支持异常分类和根因判别双任务
    4. 训练完成后绘制因果图
    """
    print("=" * 60)
    print("🧠 使用 CausalGCN_LSTM (因果感知 GCN-LSTM) 训练")
    print("=" * 60)
    
    # 1. 准备数据
    print(">>> [1/6] 加载数据集...")
    dataset = PowerGridDataset(dataset_path="dataset")
    
    # 获取时序数据
    X, Y, edge_index, edge_weight, node_mask = dataset.get_temporal_tensors(seq_len=SEQ_LEN)
    # X: [num_samples, seq_len, num_nodes, features]
    # Y: [num_samples, num_nodes]
    
    num_samples = X.shape[0]
    num_nodes = X.shape[2]
    num_features = X.shape[3]
    
    print(f">>> 数据形状: {X.shape}")
    print(f">>> 节点数: {num_nodes}, 特征数: {num_features}")
    
    # 2. 初始化设备
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        print(f">>> 检测到 CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(">>> 未检测到 CUDA，将使用 CPU 训练")
    
    # 移动图结构到设备
    edge_index_tensor = edge_index.to(device)
    edge_weight_tensor = edge_weight.to(device)
    node_mask = node_mask.to(device)
    
    # 3. 划分训练/测试集
    indices = list(range(num_samples))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, shuffle=False)
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    print(f">>> 训练集: {len(train_idx)} 样本, 测试集: {len(test_idx)} 样本")
    
    # 4. 初始化模型
    print(f">>> [2/6] 初始化 CausalGCN_LSTM 模型 (Device: {device})...")
    
    # 将 edge_index 转换为 numpy 用于模型构建
    edge_index_np = edge_index.numpy()
    
    model = CausalGCN_LSTM(
        num_nodes=num_nodes,
        num_features=num_features,
        num_classes=4,
        edge_index=edge_index_np,
        admittance_matrix=None,  # 如果有导纳矩阵可以传入
        source_node=0,  # 变压器节点（源节点）
        gcn_hidden=64,
        lstm_hidden=128,
        num_gcn_layers=2,
        num_lstm_layers=2,
        dropout=0.3
    ).to(device)
    
    # 5. 初始化因果损失函数
    print(">>> [3/6] 初始化因果损失函数...")
    
    # 类别权重 (处理不平衡)
    class_weights = torch.tensor([2.0, 50.0, 50.0, 20.0]).to(device)
    
    # 获取节点深度用于物理损失
    node_depths = model.node_depths.clone()
    
    # 创建因果损失
    criterion = create_causal_loss(
        class_weights=class_weights,
        node_depths=node_depths,
        use_focal_loss=True,
        focal_gamma=2.0,
        lambda_root=0.5,  # 根因损失权重
        lambda_sparse=0.01,  # 稀疏性损失权重
        lambda_physics=0.1  # 物理一致性损失权重
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # 6. 训练循环
    print(">>> [4/6] 开始训练...")
    loss_history = {'total': [], 'anomaly': [], 'sparse': []}
    num_epochs = 100
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_anomaly_loss = 0
        total_sparse_loss = 0
        
        # Mini-batch 训练
        perm = torch.randperm(len(train_idx))
        for i in range(0, len(train_idx), BATCH_SIZE):
            batch_indices = perm[i:i + BATCH_SIZE]
            current_batch_size = len(batch_indices)
            
            # 获取 batch 数据
            x_batch = X_train[batch_indices].to(device, non_blocking=True)
            y_batch = Y_train[batch_indices].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(x_batch, return_causal=True)
            
            # 计算因果损失
            loss, loss_dict = criterion(
                model_outputs=outputs,
                anomaly_labels=y_batch,
                root_cause_labels=None,  # 如果有根因标签可以传入
                x=x_batch,
                model=model
            )
            
            loss.backward()
            
            # 梯度裁剪
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
        
        if epoch % 10 == 0:
            # 计算当前因果矩阵的稀疏度
            causal_matrix = model.get_causal_matrix().detach().cpu().numpy()
            sparsity_ratio = np.sum(causal_matrix > 0.1) / (num_nodes * num_nodes)
            
            print(f"    Epoch {epoch:03d} | Total Loss: {avg_loss:.4f} | "
                  f"Anomaly: {avg_anomaly:.4f} | Sparse: {avg_sparse:.4f} | "
                  f"Matrix Sparsity: {sparsity_ratio*100:.1f}%")
    
    # 7. 评估
    print(">>> [5/6] 评估模型...")
    model.eval()
    correct = 0
    total = 0
    class_correct = [0, 0, 0, 0]
    class_total = [0, 0, 0, 0]
    
    test_batch_size = BATCH_SIZE * 2
    with torch.no_grad():
        for i in range(0, len(test_idx), test_batch_size):
            end_idx = min(i + test_batch_size, len(test_idx))
            batch_indices = range(i, end_idx)
            current_batch_size = len(batch_indices)
            
            x_batch = X_test[batch_indices].to(device)
            y_batch = Y_test[batch_indices].to(device)
            
            outputs = model(x_batch)
            anomaly_logits = outputs['anomaly_logits']  # [B, N, C]
            pred = anomaly_logits.argmax(dim=2)  # [B, N]
            
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
    
    acc = correct / total if total > 0 else 0
    print("=" * 50)
    print(f"✅ 最终测试集总准确率: {acc * 100:.2f}%")
    print(f"   - Class 0 (正常): {class_correct[0]}/{class_total[0]} ({100*class_correct[0]/max(class_total[0],1):.1f}%)")
    print(f"   - Class 1 (突增): {class_correct[1]}/{class_total[1]} ({100*class_correct[1]/max(class_total[1],1):.1f}%)")
    print(f"   - Class 2 (丢失): {class_correct[2]}/{class_total[2]} ({100*class_correct[2]/max(class_total[2],1):.1f}%)")
    print(f"   - Class 3 (无功): {class_correct[3]}/{class_total[3]} ({100*class_correct[3]/max(class_total[3],1):.1f}%)")
    print("=" * 50)
    
    # 保存模型
    save_dir = "result/causal_gcn_lstm"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save(model.state_dict(), f"{save_dir}/model.pth")
    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'accuracy': acc,
        'seq_len': SEQ_LEN,
        'num_nodes': num_nodes,
        'num_features': num_features,
        'num_classes': 4,
        'config': {
            'gcn_hidden': 64,
            'lstm_hidden': 128,
            'num_gcn_layers': 2,
            'num_lstm_layers': 2,
        }
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")
    print(f"\n💾 模型已保存到: {save_dir}/")
    print(f"   - 权重文件: model.pth")
    print(f"   - 完整检查点: checkpoint.pth")
    
    # 8. 绘制因果图
    print("\n>>> [6/6] 绘制因果图...")
    causal_matrix = model.get_causal_matrix().detach().cpu().numpy()
    
    # 保存因果矩阵
    np.save(f"{save_dir}/causal_matrix.npy", causal_matrix)
    print(f"   因果矩阵已保存: {save_dir}/causal_matrix.npy")
    
    # 分析因果结构
    analyze_causal_structure(causal_matrix, top_k=10)
    
    # 绘制因果图
    plot_causal_graph(causal_matrix, threshold=0.1, save_path=f"{save_dir}/causal_graph.png")
    
    # 绘制 Loss 曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(loss_history['total'], label='Total Loss', color='blue')
    axes[0].set_title("CausalGCN-LSTM Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(loss_history['anomaly'], label='Anomaly Loss', color='green')
    axes[1].set_title("Anomaly Classification Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(loss_history['sparse'], label='Sparsity Loss', color='orange')
    axes[2].set_title("Causal Sparsity Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_loss.png", dpi=300)
    plt.show()
    
    return model, acc


def train_ngc():
    """
    使用神经格兰杰因果模型 (NGC) 训练
    
    特点:
    1. 可学习的因果权重矩阵
    2. L1 稀疏性正则化
    3. 训练完成后绘制因果图
    """
    print("=" * 50)
    print("🧠 使用 Neural Granger Causality (神经格兰杰因果) 训练")
    print("=" * 50)
    
    # 1. 准备数据
    print(">>> [1/5] 加载数据集...")
    dataset = PowerGridDataset(dataset_path="dataset")
    
    # 获取时序数据
    X, Y, edge_index, edge_weight, node_mask = dataset.get_temporal_tensors(seq_len=SEQ_LEN)
    # X: [num_samples, seq_len, num_nodes, features]
    # Y: [num_samples, num_nodes]
    
    num_samples = X.shape[0]
    num_nodes = X.shape[2]
    num_features = X.shape[3]
    
    print(f">>> 数据形状: {X.shape}")
    print(f">>> 节点数: {num_nodes}, 特征数: {num_features}")
    
    # 2. 初始化设备
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        print(f">>> 检测到 CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(">>> 未检测到 CUDA，将使用 CPU 训练")
    
    # 移动图结构到 GPU
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    node_mask = node_mask.to(device)
    
    # 3. 划分训练/测试集
    indices = list(range(num_samples))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, shuffle=False)
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    print(f">>> 训练集: {len(train_idx)} 样本, 测试集: {len(test_idx)} 样本")
    
    # 4. 初始化 NGC 模型
    print(f">>> [2/5] 初始化 NGC 模型 (Device: {device})...")
    model = NeuralGrangerCausality(
        num_nodes=num_nodes,
        num_features=num_features,
        num_classes=4,
        hidden_dim=64,
        num_layers=2,
        sparsity_lambda=SPARSITY_LAMBDA
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=5e-4)
    class_weights = torch.tensor([2.0, 50.0, 50.0, 20.0]).to(device)
    
    print(f">>> 稀疏性正则化系数 λ = {SPARSITY_LAMBDA}")
    
    # 5. 训练循环
    print(">>> [3/5] 开始训练...")
    loss_history = []
    sparsity_loss_history = []
    num_epochs = 100
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_sparsity_loss = 0
        
        # Mini-batch 训练
        perm = torch.randperm(len(train_idx))
        for i in range(0, len(train_idx), BATCH_SIZE):
            batch_indices = perm[i:i + BATCH_SIZE]
            current_batch_size = len(batch_indices)
            
            # 获取 batch 数据
            x_batch = X_train[batch_indices].to(device, non_blocking=True)
            y_batch = Y_train[batch_indices].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 前向传播
            out = model(x_batch, edge_index, edge_weight)  # [B, nodes, classes]
            
            # 计算分类 Loss
            out_flat = out.view(-1, 4)  # [B*N, 4]
            y_flat = y_batch.view(-1)   # [B*N]
            mask_expanded = node_mask.repeat(current_batch_size)
            
            cls_loss = F.nll_loss(out_flat[mask_expanded], y_flat[mask_expanded], weight=class_weights)
            
            # 🔥 计算稀疏性正则化 Loss (格兰杰因果的核心)
            sparsity_loss = model.get_sparsity_loss()
            
            # 总 Loss = 分类Loss + 稀疏性Loss
            loss = cls_loss + sparsity_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += cls_loss.item()
            total_sparsity_loss += sparsity_loss.item()
        
        avg_loss = total_loss / (len(train_idx) // BATCH_SIZE + 1)
        avg_sparsity = total_sparsity_loss / (len(train_idx) // BATCH_SIZE + 1)
        loss_history.append(avg_loss)
        sparsity_loss_history.append(avg_sparsity)
        
        if epoch % 10 == 0:
            # 计算当前因果矩阵的稀疏度
            causal_matrix = model.get_causal_matrix(threshold=0.1)
            sparsity_ratio = np.sum(causal_matrix > 0) / (num_nodes * num_nodes)
            print(f"    Epoch {epoch:03d} | Cls Loss: {avg_loss:.4f} | Sparsity Loss: {avg_sparsity:.4f} | Matrix Sparsity: {sparsity_ratio*100:.1f}%")
    
    # 6. 评估
    print(">>> [4/5] 评估模型...")
    model.eval()
    correct = 0
    total = 0
    class_correct = [0, 0, 0, 0]
    class_total = [0, 0, 0, 0]
    
    test_batch_size = BATCH_SIZE * 2
    with torch.no_grad():
        for i in range(0, len(test_idx), test_batch_size):
            end_idx = min(i + test_batch_size, len(test_idx))
            batch_indices = range(i, end_idx)
            current_batch_size = len(batch_indices)
            
            x_batch = X_test[batch_indices].to(device)
            y_batch = Y_test[batch_indices].to(device)
            
            out = model(x_batch, edge_index, edge_weight)
            pred = out.argmax(dim=2)
            
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
    
    acc = correct / total
    print("=" * 40)
    print(f"✅ 最终测试集总准确率: {acc * 100:.2f}%")
    print(f"   - Class 0 (正常): {class_correct[0]}/{class_total[0]}")
    print(f"   - Class 1 (突增): {class_correct[1]}/{class_total[1]}")
    print(f"   - Class 2 (丢失): {class_correct[2]}/{class_total[2]}")
    print(f"   - Class 3 (无功): {class_correct[3]}/{class_total[3]}")
    print("=" * 40)
    
    # 保存模型
    save_dir = "result/ngc"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save(model.state_dict(), f"{save_dir}/model.pth")
    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'accuracy': acc,
        'seq_len': SEQ_LEN,
        'num_nodes': num_nodes,
        'num_features': num_features,
        'num_classes': 4,
        'sparsity_lambda': SPARSITY_LAMBDA
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")
    print(f"\n💾 模型已保存到: {save_dir}/")
    print(f"   - 权重文件: model.pth")
    print(f"   - 完整检查点: checkpoint.pth")
    
    # 7. 绘制因果图
    print("\n>>> [5/5] 绘制格兰杰因果图...")
    causal_matrix = model.get_causal_matrix(threshold=0.05)
    
    # 保存因果矩阵
    np.save(f"{save_dir}/causal_matrix.npy", causal_matrix)
    print(f"   因果矩阵已保存: {save_dir}/causal_matrix.npy")
    
    # 分析因果结构
    analyze_causal_structure(causal_matrix, top_k=10)
    
    # 绘制因果图
    plot_causal_graph(causal_matrix, threshold=0.1, save_path=f"{save_dir}/causal_graph.png")
    
    # 绘制 Loss 曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(loss_history, label='Classification Loss')
    axes[0].set_title("NGC Training Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(sparsity_loss_history, label='Sparsity Loss', color='orange')
    axes[1].set_title("Sparsity Regularization Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("L1 Loss")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_loss.png", dpi=300)
    plt.show()


def train_temporal():
    """使用时序GNN (TGCN) 训练"""
    print("=" * 50)
    print("🔥 使用 TGCN (时序图卷积网络) 训练")
    print("=" * 50)
    
    # 1. 准备数据
    print(">>> [1/4] 加载数据集...")
    dataset = PowerGridDataset(dataset_path="dataset")
    
    # 获取时序数据
    X, Y, edge_index, edge_weight, node_mask = dataset.get_temporal_tensors(seq_len=SEQ_LEN)
    # X: [num_samples, seq_len, num_nodes, features]
    # Y: [num_samples, num_nodes]
    
    num_samples = X.shape[0]
    num_features = X.shape[3]
    
    # 2. 初始化设备
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        print(f">>> 检测到 CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(">>> 未检测到 CUDA，将使用 CPU 训练")
    
    # 移动图结构到 GPU
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    node_mask = node_mask.to(device)
    
    # 3. 划分训练/测试集
    indices = list(range(num_samples))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, shuffle=False)
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    print(f">>> 训练集: {len(train_idx)} 样本, 测试集: {len(test_idx)} 样本")
    
    # 4. 初始化模型
    print(f">>> [2/4] 初始化 TGCN 模型 (Device: {device})...")
    model = TGCN(
        num_features=num_features,
        num_classes=4,
        hidden_dim=64,
        num_layers=2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=5e-4)
    class_weights = torch.tensor([2.0, 50.0, 50.0, 20.0]).to(device)
    
    # 5. 训练循环
    print(">>> [3/4] 开始训练...")
    loss_history = []
    num_epochs = 100
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Mini-batch 训练
        perm = torch.randperm(len(train_idx))
        for i in range(0, len(train_idx), BATCH_SIZE):
            batch_indices = perm[i:i + BATCH_SIZE]
            current_batch_size = len(batch_indices)
            
            # 获取 batch 数据
            x_batch = X_train[batch_indices].to(device, non_blocking=True)  # [B, seq_len, nodes, features]
            y_batch = Y_train[batch_indices].to(device, non_blocking=True)  # [B, nodes]
            
            optimizer.zero_grad()
            
            # --- 🔥 并行 Batch 处理 ---
            # 直接将整个 batch 传给模型，不再按样本循环
            out = model(x_batch, edge_index, edge_weight)  # [B, nodes, classes]
            
            # 计算 Loss (需要展平)
            out_flat = out.view(-1, 4)  # [B*N, 4]
            y_flat = y_batch.view(-1)   # [B*N]
            
            # 扩展 mask: [nodes] -> [B*nodes]
            # 注意: 所有样本共享相同的拓扑和 mask
            mask_expanded = node_mask.repeat(current_batch_size)
            
            loss = F.nll_loss(out_flat[mask_expanded], y_flat[mask_expanded], weight=class_weights)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(train_idx) // BATCH_SIZE + 1)
        loss_history.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch:03d} | Loss: {avg_loss:.4f}")
    
    # 6. 评估
    print(">>> [4/4] 评估模型...")
    model.eval()
    correct = 0
    total = 0
    class_correct = [0, 0, 0, 0]
    class_total = [0, 0, 0, 0]
    
    # 评估时也使用 Batch 处理以加速
    test_batch_size = BATCH_SIZE * 2
    with torch.no_grad():
        for i in range(0, len(test_idx), test_batch_size):
            end_idx = min(i + test_batch_size, len(test_idx))
            batch_indices = range(i, end_idx)
            current_batch_size = len(batch_indices)
            
            x_batch = X_test[batch_indices].to(device)
            y_batch = Y_test[batch_indices].to(device)
            
            out = model(x_batch, edge_index, edge_weight) # [B, N, C]
            pred = out.argmax(dim=2) # [B, N]
            
            mask_expanded = node_mask.repeat(current_batch_size) # [B*N]
            
            pred_flat = pred.view(-1)
            y_flat = y_batch.view(-1)
            
            # 统计
            valid_mask = mask_expanded
            correct += (pred_flat[valid_mask] == y_flat[valid_mask]).sum().item()
            total += valid_mask.sum().item()
            
            for c in range(4):
                c_mask = valid_mask & (y_flat == c)
                class_total[c] += c_mask.sum().item()
                class_correct[c] += (pred_flat[c_mask] == y_flat[c_mask]).sum().item()
    
    acc = correct / total
    print("=" * 40)
    print(f"✅ 最终测试集总准确率: {acc * 100:.2f}%")
    print(f"   - Class 0 (正常): {class_correct[0]}/{class_total[0]}")
    print(f"   - Class 1 (突增): {class_correct[1]}/{class_total[1]}")
    print(f"   - Class 2 (丢失): {class_correct[2]}/{class_total[2]}")
    print(f"   - Class 3 (无功): {class_correct[3]}/{class_total[3]}")
    print("=" * 40)
    
    # 保存模型
    save_dir = "result/tgcn"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save(model.state_dict(), f"{save_dir}/model.pth")
    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'accuracy': acc,
        'seq_len': SEQ_LEN,
        'num_features': num_features,
        'num_classes': 4
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")
    print(f"\n💾 模型已保存到: {save_dir}/")
    print(f"   - 权重文件: model.pth")
    print(f"   - 完整检查点: checkpoint.pth")
    
    # 画图
    plt.plot(loss_history)
    plt.title("TGCN Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"{save_dir}/training_loss.png", dpi=300)
    plt.show()


def train_gcn():
    """使用普通 GCN 训练 (保留原有逻辑)"""
    from torch_geometric.data import DataLoader
    
    print("=" * 50)
    print("使用普通 GCN 训练")
    print("=" * 50)
    # 1. 准备数据
    print(">>> [1/4] 加载数据集...")
    dataset = PowerGridDataset(dataset_path="dataset")
    data_list = dataset.get_pyg_data_list()

    # 2. 初始化设备 (GPU 优先)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        print(
            f">>> 检测到 CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}"
        )
    else:
        print(">>> 未检测到 CUDA，将使用 CPU 训练（如需 GPU，请安装 CUDA 版 PyTorch）")

    # 2. 划分训练集/测试集
    train_data, test_data = train_test_split(data_list, test_size=0.2, shuffle=False)

    # DataLoader 小优化：GPU 时启用 pin_memory，数据拷贝更快
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, pin_memory=use_cuda, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, pin_memory=use_cuda, num_workers=0)

    # 3. 初始化模型
    print(f">>> [2/4] 初始化 GCN 模型 (Device: {device})...")

    # 输入特征=3 (P,Q,V)，输出类别=4
    model = GCN(num_features=6, num_classes=4).to(device)

    # 【修改点 1】：学习率从 0.01 改为 0.005，因为加权后 Loss 会变大，步长小一点更稳
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # 【修改点 2】：定义暴力权重 (Violent Weights)
    # 告诉模型：猜错一个 Class 2 (丢失)，相当于猜错 300 个正常样本！
    # 顺序对应: [Class 0, Class 1, Class 2, Class 3]
    class_weights = torch.tensor([2.0, 50.0, 50.0, 20.0]).to(device)

    # 4. 开始训练循环
    print(">>> [3/4] 开始训练...")
    loss_history = []

    model.train()
    for epoch in range(100):  # 训练 300 轮
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=use_cuda)
            optimizer.zero_grad()

            out = model(batch)

            # 【修改点 3】：把 class_weights 传进去
            # 关键：只计算 mask=True 的节点的 Loss，并应用权重
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask], weight=class_weights)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:03d} | Loss: {avg_loss:.4f}")

    # 5. 评估与保存
    print(">>> [4/4] 评估模型...")
    model.eval()
    correct = 0
    total = 0

    # 新增：顺便统计一下各类的准确率，防止只看总分
    class_correct = [0, 0, 0, 0]
    class_total = [0, 0, 0, 0]

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device, non_blocking=use_cuda)
            pred = model(batch).argmax(dim=1)

            # 只统计有效节点
            mask = batch.train_mask

            # 总准确率
            correct += (pred[mask] == batch.y[mask]).sum().item()
            total += mask.sum().item()

            # 分类统计 (可选，方便调试)
            for c in range(4):
                # 找出 mask=True 且 label=c 的节点
                c_mask = mask & (batch.y == c)
                class_total[c] += c_mask.sum().item()
                class_correct[c] += (pred[c_mask] == batch.y[c_mask]).sum().item()

    acc = correct / total
    print("=" * 40)
    print(f"✅ 最终测试集总准确率: {acc * 100:.2f}%")
    print(f"   - Class 0 (正常): {class_correct[0]}/{class_total[0]}")
    print(f"   - Class 1 (突增): {class_correct[1]}/{class_total[1]}")
    print(f"   - Class 2 (丢失): {class_correct[2]}/{class_total[2]}")
    print(f"   - Class 3 (无功): {class_correct[3]}/{class_total[3]}")
    print("=" * 40)

    # 保存模型 (完整版，包含更多信息方便后续使用)
    save_dir = "result/gcn"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 方式1: 只保存权重 (轻量级，用于推理)
    torch.save(model.state_dict(), f"{save_dir}/model.pth")
    
    # 方式2: 保存完整检查点 (用于断点续训)
    checkpoint = {
        'epoch': 100,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'accuracy': acc,
        'class_weights': class_weights.cpu(),
        'num_features': 6,
        'num_classes': 4
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")
    print(f"\n💾 模型已保存到: {save_dir}/")
    print(f"   - 权重文件: model.pth")
    print(f"   - 完整检查点: checkpoint.pth")

    # 画图
    plt.plot(loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"{save_dir}/training_loss.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    if USE_CAUSAL_LSTM:
        train_causal_gcn_lstm()
    elif USE_NGC:
        train_ngc()
    elif USE_TEMPORAL:
        train_temporal()
    else:
        train_gcn()