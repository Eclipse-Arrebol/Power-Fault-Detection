"""
因果损失函数

包含：
1. 异常分类损失
2. 根因判别损失
3. 因果稀疏性损失
4. 物理一致性损失（KVL/KCL约束）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, 
                 gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class PhysicsConsistencyLoss(nn.Module):
    """
    物理一致性损失
    
    确保学习到的因果关系符合电网物理规律
    """
    
    def __init__(self, node_depths: torch.Tensor, 
                 delta_p_idx: int = 3,
                 delta_q_idx: int = 4, 
                 delta_v_idx: int = 5):
        """
        Args:
            node_depths: 每个节点到源的深度
            delta_p_idx: ΔP 在特征中的索引
            delta_q_idx: ΔQ 在特征中的索引
            delta_v_idx: ΔV 在特征中的索引
        """
        super().__init__()
        self.register_buffer('node_depths', node_depths)
        self.delta_p_idx = delta_p_idx
        self.delta_q_idx = delta_q_idx
        self.delta_v_idx = delta_v_idx
    
    def forward(self, causal_matrix: torch.Tensor, 
                x: torch.Tensor,
                anomaly_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            causal_matrix: (N, N) 因果矩阵
            x: (B, T, N, F) 输入特征
            anomaly_labels: (B, N) 异常标签
        
        Returns:
            loss: 物理一致性损失
        """
        batch_size, seq_len, N, num_features = x.shape
        loss = torch.tensor(0.0, device=x.device)
        
        # 1. 功率变化方向一致性
        # 过载(1): ΔP > 0, 丢失(2): ΔP < 0, 无功(3): ΔQ > 0
        delta_P = x[:, -1, :, self.delta_p_idx]  # (B, N) 最后时刻的 ΔP
        delta_Q = x[:, -1, :, self.delta_q_idx]  # (B, N)
        
        # 过载节点的 ΔP 应该 > 0
        overload_mask = (anomaly_labels == 1).float()
        overload_loss = F.relu(-delta_P) * overload_mask
        loss += overload_loss.mean()
        
        # 丢失节点的 ΔP 应该 < 0
        drop_mask = (anomaly_labels == 2).float()
        drop_loss = F.relu(delta_P) * drop_mask
        loss += drop_loss.mean()
        
        # 无功干扰节点的 ΔQ 应该 > 0
        reactive_mask = (anomaly_labels == 3).float()
        reactive_loss = F.relu(-delta_Q) * reactive_mask
        loss += reactive_loss.mean()
        
        # 2. 因果方向一致性
        # 检查强因果关系是否符合上游→下游
        # 如果 C[i,j] 很大，那么 depth[i] 应该 <= depth[j]
        depths = self.node_depths  # (N,)
        depth_diff = depths.unsqueeze(0) - depths.unsqueeze(1)  # (N, N)
        # depth_diff[i,j] = depth[i] - depth[j]
        # 如果 i 是上游（depth 小），则 depth_diff[i,j] < 0
        
        # 惩罚：下游影响上游的强因果关系
        # C[i,j] * max(0, depth[i] - depth[j])
        direction_violation = causal_matrix * F.relu(depth_diff)
        loss += direction_violation.mean() * 0.1
        
        return loss


class CausalLoss(nn.Module):
    """
    综合因果损失函数
    
    总损失 = α * 分类损失 + β * 根因损失 + γ * 稀疏性损失 + δ * 物理损失
    """
    
    def __init__(self, num_classes: int = 4,
                 class_weights: Optional[torch.Tensor] = None,
                 node_depths: Optional[torch.Tensor] = None,
                 use_focal_loss: bool = True,
                 focal_gamma: float = 2.0,
                 lambda_root: float = 1.0,
                 lambda_sparse: float = 0.01,
                 lambda_physics: float = 0.1):
        """
        Args:
            num_classes: 异常类别数
            class_weights: 类别权重（处理不平衡）
            node_depths: 节点深度
            use_focal_loss: 是否使用 Focal Loss
            focal_gamma: Focal Loss 的 gamma 参数
            lambda_root: 根因损失权重
            lambda_sparse: 稀疏性损失权重
            lambda_physics: 物理损失权重
        """
        super().__init__()
        
        self.lambda_root = lambda_root
        self.lambda_sparse = lambda_sparse
        self.lambda_physics = lambda_physics
        
        # 异常分类损失
        if use_focal_loss:
            self.anomaly_loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            self.anomaly_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        # 根因判别损失
        self.root_cause_loss_fn = nn.BCEWithLogitsLoss()
        
        # 物理一致性损失
        if node_depths is not None:
            self.physics_loss_fn = PhysicsConsistencyLoss(node_depths)
        else:
            self.physics_loss_fn = None
    
    def forward(self, model_outputs: Dict[str, torch.Tensor],
                anomaly_labels: torch.Tensor,
                root_cause_labels: Optional[torch.Tensor] = None,
                x: Optional[torch.Tensor] = None,
                model: Optional[nn.Module] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            model_outputs: 模型输出字典，包含 anomaly_logits, root_cause_logits
            anomaly_labels: (B, N) 异常标签
            root_cause_labels: (B, N) 根因标签（可选）
            x: (B, T, N, F) 输入特征（用于物理损失）
            model: 模型对象（用于获取因果矩阵）
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失明细
        """
        loss_dict = {}
        
        # 1. 异常分类损失
        anomaly_logits = model_outputs['anomaly_logits']  # (B, N, C)
        anomaly_logits_flat = anomaly_logits.view(-1, anomaly_logits.size(-1))
        anomaly_labels_flat = anomaly_labels.view(-1)
        
        loss_anomaly = self.anomaly_loss_fn(anomaly_logits_flat, anomaly_labels_flat)
        loss_dict['anomaly'] = loss_anomaly.item()
        
        total_loss = loss_anomaly
        
        # 2. 根因判别损失
        if root_cause_labels is not None and 'root_cause_logits' in model_outputs:
            root_cause_logits = model_outputs['root_cause_logits']  # (B, N)
            loss_root = self.root_cause_loss_fn(root_cause_logits, root_cause_labels.float())
            loss_dict['root_cause'] = loss_root.item()
            total_loss += self.lambda_root * loss_root
        
        # 3. 因果稀疏性损失
        if model is not None and hasattr(model, 'causal_sparsity_loss'):
            loss_sparse = model.causal_sparsity_loss()
            loss_dict['sparse'] = loss_sparse.item()
            total_loss += self.lambda_sparse * loss_sparse
        
        # 4. 物理一致性损失
        if (self.physics_loss_fn is not None and 
            x is not None and 
            model is not None and 
            hasattr(model, 'get_causal_matrix')):
            
            causal_matrix = model.get_causal_matrix()
            loss_physics = self.physics_loss_fn(causal_matrix, x, anomaly_labels)
            loss_dict['physics'] = loss_physics.item()
            total_loss += self.lambda_physics * loss_physics
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


def create_causal_loss(class_weights: Optional[torch.Tensor] = None,
                       node_depths: Optional[torch.Tensor] = None,
                       **kwargs) -> CausalLoss:
    """
    工厂函数：创建因果损失
    """
    return CausalLoss(
        class_weights=class_weights,
        node_depths=node_depths,
        **kwargs
    )
