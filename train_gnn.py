import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


# ==========================================
# 1. 数据加载与预处理类
# ==========================================
class PowerGridDataset:
    def __init__(self, dataset_path="dataset"):
        print(">>> 正在加载数据集...")

        # 1. 读取 CSV 文件
        self.p_df = pd.read_csv(f"{dataset_path}/p_mw.csv")
        self.q_df = pd.read_csv(f"{dataset_path}/q_mvar.csv")
        self.v_df = pd.read_csv(f"{dataset_path}/vm_pu.csv")
        self.labels_df = pd.read_csv(f"{dataset_path}/labels.csv")
        self.edges_df = pd.read_csv(f"{dataset_path}/edges.csv")
        self.bus_map = pd.read_csv(f"{dataset_path}/bus_map.csv")

        self.num_timesteps = self.p_df.shape[0]
        self.num_buses = self.v_df.shape[1]  # 电压数据的列数等于节点数

        # 2. 构建图结构 (Edge Index)
        # GNN 需要 [2, Num_Edges] 的张量
        source = self.edges_df['from_bus'].values
        target = self.edges_df['to_bus'].values
        # 转为双向图 (因为电力影响是相互的)
        edge_index = torch.tensor([
            np.concatenate([source, target]),
            np.concatenate([target, source])
        ], dtype=torch.long)
        self.edge_index = edge_index

        # 3. 数据映射与特征构建
        # 我们需要构建 (Time, Buses, Features) 的矩阵
        # Features = [P, Q, V] (3个特征)
        print(">>> 正在进行 [Load -> Bus] 数据映射...")
        self.features = np.zeros((self.num_timesteps, self.num_buses, 3))
        self.labels = np.zeros((self.num_timesteps, self.num_buses))

        # 创建掩码 (Mask)：记录哪些节点是有负载的 (我们需要对这些节点计算Loss)
        # 没有负载的节点(如中间连接点)不需要预测异常，或者标记为正常
        self.node_mask = np.zeros(self.num_buses, dtype=bool)

        # 填入电压 (V) - 所有节点都有电压
        self.features[:, :, 2] = self.v_df.values

        # 填入功率 (P, Q) 和 标签 - 需要查表 bus_map
        # bus_map 结构: [name, bus] -> load_0 在 bus_3
        load_to_bus = dict(zip(self.bus_map.index, self.bus_map['bus']))

        for load_idx in range(self.p_df.shape[1]):
            # 找到这个负载挂在哪个节点上
            bus_idx = load_to_bus[load_idx]
            self.node_mask[bus_idx] = True  # 标记该节点有效

            # 填入数据
            self.features[:, bus_idx, 0] = self.p_df.iloc[:, load_idx].values  # P
            self.features[:, bus_idx, 1] = self.q_df.iloc[:, load_idx].values  # Q
            self.labels[:, bus_idx] = self.labels_df.iloc[:, load_idx].values

        # 4. 归一化 (非常重要！P/Q 很小，V 很大)
        scaler = StandardScaler()
        # 展平 -> 归一化 -> 变回来
        flat_features = self.features.reshape(-1, 3)
        flat_features = scaler.fit_transform(flat_features)
        self.features = flat_features.reshape(self.num_timesteps, self.num_buses, 3)

        print(">>> 数据预处理完成！")

    def get_pyg_data_list(self):
        """将每个时间步转换为一个 PyG Data 对象"""
        data_list = []
        for t in range(self.num_timesteps):
            # 特征: [Num_Buses, 3]
            x = torch.tensor(self.features[t], dtype=torch.float)
            # 标签: [Num_Buses]
            y = torch.tensor(self.labels[t], dtype=torch.long)
            # 掩码: [Num_Buses]
            mask = torch.tensor(self.node_mask, dtype=torch.bool)

            data = Data(x=x, edge_index=self.edge_index, y=y)
            data.train_mask = mask  # 只在有负载的节点上训练
            data_list.append(data)
        return data_list


# ==========================================
# 2. 定义 GCN 模型
# ==========================================
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        # 定义两层 GCN
        # 输入维度: 3 (P, Q, V)
        # 隐藏层: 64
        # 输出维度: 4 (Normal, Spike, Drop, HighQ)
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 第一层卷积 + ReLU激活 + Dropout防过拟合
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 第二层卷积
        x = self.conv2(x, edge_index)

        # 输出 LogSoftmax (用于分类)
        return F.log_softmax(x, dim=1)


# ==========================================
# 3. 训练主流程
# ==========================================
def train_model():
    # 1. 准备数据
    dataset = PowerGridDataset()
    data_list = dataset.get_pyg_data_list()

    # 划分训练集和测试集 (按时间切分，前80%训练，后20%测试)
    train_data, test_data = train_test_split(data_list, test_size=0.2, shuffle=False)

    # 使用 DataLoader 批处理 (batch_size=16 代表一次训练16个时间步)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    # 2. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features=3, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print(f">>> 开始训练 (Device: {device})...")

    # 记录 Loss 用于画图
    loss_history = []

    # 3. 训练循环
    model.train()
    for epoch in range(100):  # 训练 100 轮
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # 前向传播
            out = model(batch)

            # 计算 Loss
            # 注意：我们只计算 mask=True (有负载) 的节点的 Loss
            # batch.y 是标签，batch.train_mask 是掩码
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])

            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {avg_loss:.4f}")

    # 4. 测试/评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch).argmax(dim=1)

            # 只统计 mask 部分的准确率
            mask = batch.train_mask
            correct += (pred[mask] == batch.y[mask]).sum().item()
            total += mask.sum().item()

    acc = correct / total
    print("=" * 30)
    print(f"✅ 训练完成！测试集准确率: {acc:.4f}")
    print("=" * 30)

    # 5. 画 Loss 曲线
    plt.plot(loss_history)
    plt.title("GCN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # 6. 保存模型
    torch.save(model.state_dict(), "gcn_model.pth")
    print("模型已保存为 gcn_model.pth")


if __name__ == "__main__":
    train_model()