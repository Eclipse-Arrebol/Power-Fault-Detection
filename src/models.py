import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()

        # 1. 特征编码层 (先用 MLP 提取单点特征)
        # 把输入的 3 个特征 (P, Q, V) 映射到 64 维
        self.lin_in = torch.nn.Linear(num_features, 64)

        # 2. 图卷积层 (加深到 3 层，扩大感受野)
        self.conv1 = GCNConv(64, 128)
        self.bn1 = BatchNorm(128)  # 加个 BatchNorm 防止训练不稳定

        self.conv2 = GCNConv(128, 128)
        self.bn2 = BatchNorm(128)

        self.conv3 = GCNConv(128, 64)
        self.bn3 = BatchNorm(64)

        # 3. 输出分类层
        self.lin_out = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # --- 编码阶段 ---
        x = self.lin_in(x)
        x = F.relu(x)

        # --- 图卷积阶段 1 ---
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)  # 这里的 dropout 调小一点

        # --- 图卷积阶段 2 ---
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        # --- 图卷积阶段 3 ---
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # --- 输出阶段 ---
        x = self.lin_out(x)

        return F.log_softmax(x, dim=1)