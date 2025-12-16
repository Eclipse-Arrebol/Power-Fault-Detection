import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        # 定义网络结构
        # 输入维度: 3 (P, Q, V)
        # 隐藏层: 64
        # 输出维度: 4 (0=正常, 1=突增, 2=掉线, 3=高无功)
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 第一层卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 第二层卷积
        x = self.conv2(x, edge_index)

        # 输出分类概率 (LogSoftmax)
        return F.log_softmax(x, dim=1)