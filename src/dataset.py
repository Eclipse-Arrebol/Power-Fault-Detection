import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


class PowerGridDataset:
    def __init__(self, dataset_path="dataset"):
        # 1. 读取 CSV 文件
        try:
            self.p_df = pd.read_csv(f"{dataset_path}/p_mw.csv")
            self.q_df = pd.read_csv(f"{dataset_path}/q_mvar.csv")
            self.v_df = pd.read_csv(f"{dataset_path}/vm_pu.csv")
            self.labels_df = pd.read_csv(f"{dataset_path}/labels.csv")
            self.edges_df = pd.read_csv(f"{dataset_path}/edges.csv")
            self.bus_map = pd.read_csv(f"{dataset_path}/bus_map.csv")
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到数据集文件，请先运行 main.py 生成数据！")

        self.num_timesteps = self.p_df.shape[0]
        self.num_buses = self.v_df.shape[1]

        # 2. 构建图结构 (Edge Index) - 转为双向图
        source = self.edges_df['from_bus'].values
        target = self.edges_df['to_bus'].values
        self.edge_index = torch.tensor([
            np.concatenate([source, target]),
            np.concatenate([target, source])
        ], dtype=torch.long)

        # 3. 数据映射与特征构建
        # 我们构建 (Time, Buses, 3) 的矩阵 -> [P, Q, V]
        self.features = np.zeros((self.num_timesteps, self.num_buses, 3))
        self.labels = np.zeros((self.num_timesteps, self.num_buses))
        self.node_mask = np.zeros(self.num_buses, dtype=bool)  # 掩码：记录哪些节点有负载

        # 填入电压 (V)
        self.features[:, :, 2] = self.v_df.values

        # 填入功率 (P, Q) 和 标签 (需要查表映射)
        load_to_bus = dict(zip(self.bus_map.index, self.bus_map['bus']))

        for load_idx in range(self.p_df.shape[1]):
            if load_idx in load_to_bus:
                bus_idx = load_to_bus[load_idx]
                self.node_mask[bus_idx] = True
                self.features[:, bus_idx, 0] = self.p_df.iloc[:, load_idx].values
                self.features[:, bus_idx, 1] = self.q_df.iloc[:, load_idx].values
                self.labels[:, bus_idx] = self.labels_df.iloc[:, load_idx].values

        # 4. 归一化 (StandardScaler)
        # 展平 -> 归一化 -> 变回原形状
        scaler = StandardScaler()
        flat_features = self.features.reshape(-1, 3)
        flat_features = scaler.fit_transform(flat_features)
        self.features = flat_features.reshape(self.num_timesteps, self.num_buses, 3)

    def get_pyg_data_list(self):
        """生成 PyTorch Geometric 所需的数据列表"""
        data_list = []
        for t in range(self.num_timesteps):
            x = torch.tensor(self.features[t], dtype=torch.float)
            y = torch.tensor(self.labels[t], dtype=torch.long)
            mask = torch.tensor(self.node_mask, dtype=torch.bool)

            data = Data(x=x, edge_index=self.edge_index, y=y)
            data.train_mask = mask  # 重要：告诉模型只在有负载的节点计算 Loss
            data_list.append(data)
        return data_list
    