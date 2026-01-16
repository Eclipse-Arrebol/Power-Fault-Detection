import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
import numpy as np
from typing import Optional, Tuple, Dict
from collections import deque


# ============================================================
# 🔧 辅助函数：构建因果掩码
# ============================================================
def build_causal_masks(edge_index: np.ndarray, num_nodes: int, 
                       source_node: int = 0,
                       admittance_matrix: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
    """
    构建因果约束掩码

    Args:
        edge_index: (2, E) 边索引数组
        num_nodes: 节点总数
        source_node: 源节点（变压器/馈线起点）
        admittance_matrix: 导纳矩阵（可选，用于初始化因果强度）

    Returns:
        dict: {
            'adj_mask': (N, N) 邻接掩码，只有物理相邻的节点才有因果关系
            'direction_mask': (N, N) 方向掩码，因果只能从上游流向下游
            'initial_causal': (N, N) 初始因果强度（基于导纳）
            'node_depths': (N,) 每个节点到源的深度
        }
    """
    # 1. 构建邻接表
    adj_list = {i: set() for i in range(num_nodes)}
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        if src < num_nodes and dst < num_nodes:
            adj_list[src].add(dst)
            adj_list[dst].add(src)

    # 2. BFS 计算节点深度（距离源节点的跳数）
    node_depths = torch.full((num_nodes,), float('inf'))
    
    # 确保源节点在有效范围内
    if source_node >= num_nodes:
        source_node = 0
    
    node_depths[source_node] = 0
    queue = deque([source_node])
    visited = {source_node}

    while queue:
        node = queue.popleft()
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                node_depths[neighbor] = node_depths[node] + 1
                queue.append(neighbor)

    # 处理未连接的节点（设置为最大深度）
    max_depth = node_depths[node_depths != float('inf')].max() if (node_depths != float('inf')).any() else 0
    node_depths[node_depths == float('inf')] = max_depth + 1

    # 3. 构建邻接掩码（1 表示有边相连）
    adj_mask = torch.zeros(num_nodes, num_nodes)
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        if src < num_nodes and dst < num_nodes:
            adj_mask[src, dst] = 1.0
            adj_mask[dst, src] = 1.0
    
    # 添加自环
    adj_mask += torch.eye(num_nodes)
    adj_mask = torch.clamp(adj_mask, 0, 1)

    # 4. 构建方向掩码（只有上游→下游或同层才允许）
    # direction_mask[i, j] = 1 表示 j 可以影响 i（j 的深度 <= i 的深度）
    direction_mask = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if node_depths[j] <= node_depths[i]:
                direction_mask[i, j] = 1.0

    # 5. 初始化因果强度（基于导纳或均匀分布）
    if admittance_matrix is not None and admittance_matrix.shape == (num_nodes, num_nodes):
        # 使用导纳作为初始因果强度
        initial_causal = torch.tensor(admittance_matrix, dtype=torch.float32)
        # 归一化
        initial_causal = initial_causal / (initial_causal.max() + 1e-8)
    else:
        # 均匀初始化
        initial_causal = torch.ones(num_nodes, num_nodes) / num_nodes

    # 应用掩码
    initial_causal = initial_causal * adj_mask * direction_mask

    return {
        'adj_mask': adj_mask,
        'direction_mask': direction_mask,
        'initial_causal': initial_causal,
        'node_depths': node_depths.float()
    }


# ============================================================
# 🔥 传统GCN代码
# ============================================================
class GCN(torch.nn.Module):
    """
    普通图卷积网络 - 用于单时间步预测（保留兼容性）
    """
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.lin_in = torch.nn.Linear(num_features, 64)
        self.conv1 = GCNConv(64, 128)
        self.bn1 = BatchNorm(128)
        self.conv2 = GCNConv(128, 128)
        self.bn2 = BatchNorm(128)
        self.conv3 = GCNConv(128, 64)
        self.bn3 = BatchNorm(64)
        self.lin_out = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None

        x = self.lin_in(x)
        x = F.relu(x)

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.lin_out(x)
        return F.log_softmax(x, dim=1)



# ============================================================
# 🔥 TGCN: Temporal Graph Convolutional Network
# ============================================================
def batch_static_graph(edge_index, edge_weight, batch_size, num_nodes):
    """
    辅助函数: 将静态图扩展为 batch 图 (Block Diagonal)
    [2, E] -> [2, E * B]
    """
    num_edges = edge_index.size(1)
    device = edge_index.device
    
    # 生成偏移量: [0, N, 2N, ..., (B-1)N]
    shift = torch.arange(batch_size, device=device).view(-1, 1) * num_nodes
    shift = shift.repeat(1, num_edges).view(-1)
    
    # 复制边索引并加上偏移
    batched_edge_index = edge_index.repeat(1, batch_size)  # [2, B*E]
    batched_edge_index += shift
    
    # 复制边权重
    batched_edge_weight = edge_weight.repeat(batch_size) if edge_weight is not None else None
    
    return batched_edge_index, batched_edge_weight
class TGCNCell(nn.Module):
    """
    TGCN 单元: GCN + GRU 的融合
    - GCN 负责空间特征提取 (捕捉节点间电气耦合关系)
    - GRU 负责时间序列建模 (捕捉负荷变化的时序模式)
    
    参考论文: T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction
    """
    def __init__(self, in_channels, out_channels):
        super(TGCNCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # GCN 用于提取空间特征 (用于 GRU 的 reset/update gate)
        self.graph_conv1 = GCNConv(in_channels + out_channels, out_channels)
        self.graph_conv2 = GCNConv(in_channels + out_channels, out_channels)
        self.graph_conv3 = GCNConv(in_channels + out_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight, h):
        combined = torch.cat([x, h], dim=1)
        
        # Reset gate
        r = torch.sigmoid(self.graph_conv1(combined, edge_index, edge_weight))
        # Update gate
        u = torch.sigmoid(self.graph_conv2(combined, edge_index, edge_weight))
        # 候选隐状态
        combined_r = torch.cat([x, r * h], dim=1)
        c = torch.tanh(self.graph_conv3(combined_r, edge_index, edge_weight))
        # 最终隐状态
        h_new = u * h + (1 - u) * c
        
        return h_new
class TGCN(nn.Module):
    """
    时序图卷积网络 (Batch 并行版)
    
    架构:
    1. 输入编码: Linear 把原始特征映射到隐藏维度
    2. TGCN Cell: 多个时间步共享同一个 TGCN Cell，逐步更新隐状态
    3. 输出层: 用最后时刻的隐状态做分类
    
    适用于: 电网负荷异常检测，需要同时考虑空间拓扑和时间演变
    """
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=2):
        super(TGCN, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lin_in = nn.Linear(num_features, hidden_dim)
        
        self.tgcn_cells = nn.ModuleList()
        for i in range(num_layers):
            self.tgcn_cells.append(TGCNCell(hidden_dim, hidden_dim))
        
        self.bn = BatchNorm(hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x_seq, edge_index, edge_weight, node_mask=None):
        """
        Args:
            x_seq: [Batch, Seq, Nodes, Features] 或 [Seq, Nodes, Features]
        """
        # 处理 Batch 维度
        if x_seq.dim() == 4:
            batch_size, seq_len, num_nodes, _ = x_seq.shape
            
            # 1. 扩展图结构 (Batch Graph)
            edge_index, edge_weight = batch_static_graph(edge_index, edge_weight, batch_size, num_nodes)
            
            # 2. 变形数据: [Batch, Seq, Nodes, F] -> [Seq, Batch*Nodes, F]
            # 这样 GCN 就可以把 (Batch*Nodes) 当作一个超大图的所有节点一次性处理
            x_seq = x_seq.permute(1, 0, 2, 3).reshape(seq_len, batch_size * num_nodes, -1)
            
            total_nodes = batch_size * num_nodes
        else:
            # 单样本情况 (保留兼容性)
            seq_len, num_nodes, _ = x_seq.shape
            batch_size = 1
            total_nodes = num_nodes
        
        device = x_seq.device
        
        # 初始化隐状态 [Total_Nodes, Hidden]
        h_list = [
            torch.zeros(total_nodes, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]
        
        # 逐时间步处理 (但并行的 Batch)
        for t in range(seq_len):
            x_t = x_seq[t]  # [Total_Nodes, Features]
            
            x_t = self.lin_in(x_t)
            x_t = F.relu(x_t)
            
            for layer_idx, cell in enumerate(self.tgcn_cells):
                h_list[layer_idx] = cell(x_t, edge_index, edge_weight, h_list[layer_idx])
                x_t = h_list[layer_idx]
                if layer_idx < self.num_layers - 1:
                    x_t = self.dropout(x_t)
        
        # Output reshape: [Total_Nodes, Hidden]
        h_final = h_list[-1]
        h_final = self.bn(h_final)
        h_final = F.relu(h_final)
        
        out = self.lin_out(h_final) # [Total_Nodes, Classes]
        
        # 还原形状: [Batch*Nodes, C] -> [Batch, Nodes, C]
        if batch_size > 1:
            out = out.view(batch_size, num_nodes, -1)
            
        return F.log_softmax(out, dim=-1)


# ============================================================
# 🔥 NGC格兰杰因果模型
# 🔥 Neural Granger Causality (NGC) - 神经格兰杰因果模型
# ============================================================
class NeuralGrangerCausality(nn.Module):
    """
    神经格兰杰因果模型 - 用于发现时序数据中的因果关系
    
    原理:
    - 使用可学习的因果权重矩阵 W，W[i,j] 表示节点 j 对节点 i 的因果影响强度
    - 通过 L1 稀疏性正则化，自动发现稀疏的因果结构
    - 结合 GCN 进行空间特征提取，GRU 进行时间序列建模
    
    Loss = 预测Loss + λ * ||W||_1 (稀疏性约束)
    
    参考: Neural Granger Causality (Tank et al., 2021)
    """
    def __init__(self, num_nodes, num_features, num_classes, hidden_dim=64, 
                 num_layers=2, sparsity_lambda=0.01):
        super(NeuralGrangerCausality, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sparsity_lambda = sparsity_lambda
        
        # 🔥 核心: 可学习的因果权重矩阵 W [num_nodes, num_nodes]
        # W[i, j] 表示节点 j 对节点 i 的格兰杰因果影响
        # 初始化为小的随机值，让网络自己学习稀疏结构
        self.causal_weight = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.01)
        
        # 输入编码层
        self.lin_in = nn.Linear(num_features, hidden_dim)
        
        # 因果注意力融合层 - 将因果权重融入特征
        self.causal_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # GCN 层用于空间特征提取
        self.gcn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.bn_layers.append(BatchNorm(hidden_dim))
        
        # GRU 用于时间序列建模
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=False,
            dropout=0.2
        )
        
        # 输出层
        self.bn_out = BatchNorm(hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def get_causal_matrix(self, threshold=0.1):
        """
        获取因果矩阵 (经过阈值处理)
        threshold: 小于该值的因果权重被视为无因果关系
        """
        with torch.no_grad():
            W = torch.abs(self.causal_weight)
            # 归一化到 [0, 1]
            W = W / (W.max() + 1e-8)
            # 阈值处理
            W = torch.where(W > threshold, W, torch.zeros_like(W))
            return W.cpu().numpy()
    
    def get_sparsity_loss(self):
        """
        计算稀疏性正则化 Loss (L1 范数)
        """
        return self.sparsity_lambda * torch.sum(torch.abs(self.causal_weight))
    
    def apply_causal_attention(self, x):
        """
        应用因果注意力机制
        Args:
            x: [num_nodes, hidden_dim]
        Returns:
            x_causal: [num_nodes, hidden_dim] 融合了因果信息的特征
        """
        # 获取因果权重的 softmax (使得每个节点的输入权重和为1)
        W = torch.softmax(self.causal_weight, dim=1)  # [N, N]
        
        # 通过因果权重聚合邻居信息
        # x_agg[i] = sum_j W[i,j] * x[j] 
        x_agg = torch.matmul(W, x)  # [N, hidden]
        
        # 融合原始特征和因果聚合特征
        x_concat = torch.cat([x, x_agg], dim=-1)  # [N, 2*hidden]
        x_causal = self.causal_fusion(x_concat)  # [N, hidden]
        
        return F.relu(x_causal)
        
    def forward(self, x_seq, edge_index, edge_weight, node_mask=None):
        """
        Args:
            x_seq: [seq_len, num_nodes, features] 或 [batch, seq_len, num_nodes, features]
        Returns:
            out: [num_nodes, num_classes] 或 [batch, num_nodes, num_classes]
        """
        # 处理 batch 维度
        if x_seq.dim() == 4:
            batch_size, seq_len, num_nodes, _ = x_seq.shape
            # 逐样本处理 (因为因果矩阵是固定的)
            outputs = []
            for b in range(batch_size):
                out_b = self._forward_single(x_seq[b], edge_index, edge_weight)
                outputs.append(out_b)
            return torch.stack(outputs, dim=0)  # [B, N, C]
        else:
            return self._forward_single(x_seq, edge_index, edge_weight)
    
    def _forward_single(self, x_seq, edge_index, edge_weight):
        """
        单样本前向传播
        Args:
            x_seq: [seq_len, num_nodes, features]
        """
        seq_len, num_nodes, _ = x_seq.shape
        
        outputs = []
        for t in range(seq_len):
            x_t = x_seq[t]  # [N, F]
            
            # 1. 输入编码
            x_t = self.lin_in(x_t)
            x_t = F.relu(x_t)
            
            # 2. 应用因果注意力
            x_t = self.apply_causal_attention(x_t)
            
            # 3. GCN 空间特征提取
            for i, (gcn, bn) in enumerate(zip(self.gcn_layers, self.bn_layers)):
                x_t = gcn(x_t, edge_index, edge_weight)
                x_t = bn(x_t)
                x_t = F.relu(x_t)
                if i < self.num_layers - 1:
                    x_t = self.dropout(x_t)
            
            outputs.append(x_t)
        
        # 4. GRU 时间序列建模
        gcn_seq = torch.stack(outputs, dim=0)  # [seq_len, N, hidden]
        gru_out, _ = self.gru(gcn_seq)
        h_final = gru_out[-1]  # [N, hidden]
        
        # 5. 输出层
        h_final = self.bn_out(h_final)
        h_final = F.relu(h_final)
        out = self.lin_out(h_final)  # [N, num_classes]
        
        return F.log_softmax(out, dim=-1)


# ============================================================
# 🔥 时序GCN
# ============================================================
class TemporalGCN(nn.Module):
    """
    简化版时序GNN: GCN + GRU (更模块化，易于理解)
    """
    def __init__(self, num_features, num_classes, hidden_dim=64, gru_layers=2):
        super(TemporalGCN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.lin_in = nn.Linear(num_features, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=False,
            dropout=0.2 if gru_layers > 1 else 0
        )
        
        self.lin_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x_seq, edge_index, edge_weight, node_mask=None):
        """
        Args:
            x_seq: 时间序列 [seq_len, num_nodes, num_features]
        Returns:
            out: [num_nodes, num_classes]
        """
        seq_len, num_nodes, _ = x_seq.shape
        
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x_seq[t]
            x_t = self.lin_in(x_t)
            x_t = F.relu(x_t)
            x_t = self.conv1(x_t, edge_index, edge_weight)
            x_t = self.bn1(x_t)
            x_t = F.relu(x_t)
            x_t = self.dropout(x_t)
            x_t = self.conv2(x_t, edge_index, edge_weight)
            x_t = self.bn2(x_t)
            x_t = F.relu(x_t)
            gcn_outputs.append(x_t)
        
        gcn_seq = torch.stack(gcn_outputs, dim=0)
        gru_out, _ = self.gru(gcn_seq)
        h_final = gru_out[-1]
        
        out = self.lin_out(h_final)
        return F.log_softmax(out, dim=1)



# ============================================================
# 🔥 因果+lstm  CausalGCN_LSTM
# ============================================================
class CausalAttention(nn.Module):
    """
    因果注意力层

    学习节点间的因果影响强度，受物理约束
    """

    def __init__(self, input_dim: int, num_nodes: int,
                 adj_mask: torch.Tensor,
                 direction_mask: torch.Tensor,
                 initial_causal: Optional[torch.Tensor] = None):
        super().__init__()

        self.num_nodes = num_nodes

        # 注册掩码为 buffer（不参与梯度更新）
        self.register_buffer('adj_mask', adj_mask)
        self.register_buffer('direction_mask', direction_mask)

        # 可学习的因果强度矩阵
        if initial_causal is not None:
            self.causal_logits = nn.Parameter(initial_causal.clone())
        else:
            self.causal_logits = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        # 特征变换
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        self.scale = input_dim ** 0.5

    def get_causal_matrix(self) -> torch.Tensor:
        """
        获取有效因果矩阵 = softmax(学习到的强度) × 物理约束
        """
        # Softmax 归一化
        C = torch.softmax(self.causal_logits, dim=-1)

        # 应用物理约束
        C = C * self.adj_mask * self.direction_mask

        # 重新归一化
        row_sum = C.sum(dim=-1, keepdim=True)
        row_sum = torch.clamp(row_sum, min=1e-8)
        C = C / row_sum

        return C

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, num_nodes, features)

        Returns:
            output: (batch, num_nodes, features) 因果聚合后的特征
            causal_matrix: (num_nodes, num_nodes) 因果矩阵
        """
        batch_size = x.size(0)

        # 获取因果矩阵
        C = self.get_causal_matrix()  # (N, N)

        # Query, Key, Value
        Q = self.query(x)  # (B, N, F)
        K = self.key(x)  # (B, N, F)
        V = self.value(x)  # (B, N, F)

        # 注意力分数 = Q @ K^T / sqrt(d)
        attn_scores = torch.bmm(Q, K.transpose(-2, -1)) / self.scale  # (B, N, N)

        # 融合学习到的因果矩阵
        # 因果矩阵作为先验，调制注意力
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, N, N)
        attn_weights = attn_weights * C.unsqueeze(0)  # 应用因果约束

        # 重新归一化
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 聚合
        output = torch.bmm(attn_weights, V)  # (B, N, F)

        return output, C


class PhysicsGuidedCausalGCN(nn.Module):
    """
    物理引导的因果图卷积网络

    特点：
    1. 用导纳矩阵初始化因果强度
    2. 邻接掩码：只有物理相邻节点有因果关系
    3. 方向掩码：因果只能从上游传到下游
    4. 输出因果矩阵，可解释
    """

    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int,
                 edge_index: np.ndarray,
                 admittance_matrix: Optional[np.ndarray] = None,
                 source_node: int = 105,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # 构建因果掩码
        masks = build_causal_masks(
            edge_index, num_nodes, source_node, admittance_matrix
        )

        # 输入变换
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 因果注意力层
        self.causal_attention = CausalAttention(
            hidden_dim, num_nodes,
            masks['adj_mask'],
            masks['direction_mask'],
            masks['initial_causal']
        )

        # GCN 层
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        # 融合层
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 保存节点深度
        self.register_buffer('node_depths', masks['node_depths'])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, num_nodes, features)
            edge_index: (2, E)

        Returns:
            h: (batch, num_nodes, hidden_dim)
            causal_matrix: (num_nodes, num_nodes)
        """
        batch_size = x.size(0)

        # 输入投影
        h = self.input_proj(x)  # (B, N, H)
        h = F.relu(h)

        # 因果注意力聚合
        h_causal, causal_matrix = self.causal_attention(h)  # (B, N, H), (N, N)

        # GCN 聚合（逐 batch 处理）
        h_gcn_list = []
        for b in range(batch_size):
            h_b = h[b]  # (N, H)
            for gcn in self.gcn_layers:
                h_b = gcn(h_b, edge_index)
                h_b = F.relu(h_b)
                h_b = self.dropout(h_b)
            h_gcn_list.append(h_b)

        h_gcn = torch.stack(h_gcn_list, dim=0)  # (B, N, H)

        # 融合因果聚合和 GCN 聚合
        h_combined = torch.cat([h_causal, h_gcn], dim=-1)  # (B, N, 2H)
        h = self.fusion(h_combined)  # (B, N, H)
        h = self.layer_norm(h)

        return h, causal_matrix


class CausalGCN_LSTM(nn.Module):
    """
    因果感知的 GCN-LSTM 模型

    架构：
    ┌─────────────────────────────────────────────────────────┐
    │  输入: (batch, seq_len, num_nodes, features)            │
    │                      ↓                                  │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │  PhysicsGuidedCausalGCN (每个时间步)             │   │
    │  │  • 因果注意力 + GCN 聚合                         │   │
    │  │  • 输出因果矩阵                                  │   │
    │  └─────────────────────────────────────────────────┘   │
    │                      ↓                                  │
    │  ┌─────────────────────────────────────────────────┐   │
    │  │  LSTM (时序建模)                                 │   │
    │  └─────────────────────────────────────────────────┘   │
    │                      ↓                                  │
    │  ┌──────────────────┬──────────────────────────────┐   │
    │  │  异常分类头       │  根因判别头                   │   │
    │  │  (0/1/2/3)       │  (是否是故障源)               │   │
    │  └──────────────────┴──────────────────────────────┘   │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, num_nodes: int, num_features: int, num_classes: int,
                 edge_index: np.ndarray,
                 admittance_matrix: Optional[np.ndarray] = None,
                 source_node: int = 105,
                 gcn_hidden: int = 64,
                 lstm_hidden: int = 128,
                 num_gcn_layers: int = 2,
                 num_lstm_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_classes = num_classes

        # 因果 GCN
        self.causal_gcn = PhysicsGuidedCausalGCN(
            num_nodes=num_nodes,
            input_dim=num_features,
            hidden_dim=gcn_hidden,
            edge_index=edge_index,
            admittance_matrix=admittance_matrix,
            source_node=source_node,
            num_layers=num_gcn_layers,
            dropout=dropout
        )

        # LSTM 时序建模
        self.lstm = nn.LSTM(
            input_size=gcn_hidden,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )

        lstm_output_dim = lstm_hidden * 2  # 双向

        # 任务1: 异常分类头
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )

        # 任务2: 根因判别头
        # 结合节点深度信息（上游节点更可能是根因）
        self.root_cause_head = nn.Sequential(
            nn.Linear(lstm_output_dim + 1, lstm_output_dim // 2),  # +1 for depth
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, 1)
        )

        # 保存边索引
        self.register_buffer('edge_index', torch.LongTensor(edge_index))
        self.register_buffer('node_depths', self.causal_gcn.node_depths)

    def forward(self, x: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                return_causal: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, num_nodes, num_features)
            edge_index: (2, E) 可选，默认使用初始化时的
            return_causal: 是否返回因果矩阵

        Returns:
            dict: {
                'anomaly_logits': (batch, num_nodes, num_classes),
                'root_cause_logits': (batch, num_nodes),
                'causal_matrix': (num_nodes, num_nodes) [if return_causal]
            }
        """
        batch_size, seq_len, N, F = x.shape

        if edge_index is None:
            edge_index = self.edge_index

        # 对每个时间步进行因果 GCN
        h_seq = []
        causal_matrix = None

        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (B, N, F)
            h_t, causal_matrix = self.causal_gcn(x_t, edge_index)  # (B, N, H)
            h_seq.append(h_t)

        # 堆叠时间维度
        h_seq = torch.stack(h_seq, dim=1)  # (B, T, N, H)

        # LSTM 处理每个节点的时序
        # 重排为 (B*N, T, H)
        h_seq = h_seq.permute(0, 2, 1, 3).contiguous()  # (B, N, T, H)
        h_seq = h_seq.view(batch_size * N, seq_len, -1)  # (B*N, T, H)

        lstm_out, _ = self.lstm(h_seq)  # (B*N, T, 2*lstm_hidden)

        # 取最后时间步
        h_final = lstm_out[:, -1, :]  # (B*N, 2*lstm_hidden)
        h_final = h_final.view(batch_size, N, -1)  # (B, N, 2*lstm_hidden)

        # 任务1: 异常分类
        anomaly_logits = self.anomaly_classifier(h_final)  # (B, N, num_classes)

        # 任务2: 根因判别
        # 添加节点深度特征（归一化）
        depth_feat = self.node_depths.unsqueeze(0).expand(batch_size, -1)  # (B, N)
        depth_feat = depth_feat / (depth_feat.max() + 1e-8)  # 归一化
        depth_feat = depth_feat.unsqueeze(-1)  # (B, N, 1)

        h_with_depth = torch.cat([h_final, depth_feat], dim=-1)  # (B, N, 2*lstm_hidden+1)
        root_cause_logits = self.root_cause_head(h_with_depth).squeeze(-1)  # (B, N)

        outputs = {
            'anomaly_logits': anomaly_logits,
            'root_cause_logits': root_cause_logits
        }

        if return_causal:
            outputs['causal_matrix'] = causal_matrix

        return outputs

    def get_causal_matrix(self) -> torch.Tensor:
        """获取当前的因果矩阵"""
        return self.causal_gcn.causal_attention.get_causal_matrix()

    def causal_sparsity_loss(self) -> torch.Tensor:
        """因果稀疏性损失"""
        C = self.get_causal_matrix()
        return torch.mean(torch.abs(C))


def create_causal_model(num_nodes: int, num_features: int, num_classes: int,
                        edge_index: np.ndarray,
                        admittance_matrix: Optional[np.ndarray] = None,
                        source_node: int = 105,
                        **kwargs) -> CausalGCN_LSTM:
    """
    工厂函数：创建因果 GCN-LSTM 模型
    """
    return CausalGCN_LSTM(
        num_nodes=num_nodes,
        num_features=num_features,
        num_classes=num_classes,
        edge_index=edge_index,
        admittance_matrix=admittance_matrix,
        source_node=source_node,
        **kwargs
    )
