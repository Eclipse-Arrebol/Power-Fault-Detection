import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


class PowerGridDataset:
    def __init__(self, dataset_path="dataset"):
        # 1. 读取 CSV (代码不变)
        try:
            self.p_df = pd.read_csv(f"{dataset_path}/p_mw.csv")
            self.q_df = pd.read_csv(f"{dataset_path}/q_mvar.csv")
            self.v_df = pd.read_csv(f"{dataset_path}/vm_pu.csv")
            self.labels_df = pd.read_csv(f"{dataset_path}/labels.csv")
            self.edges_df = pd.read_csv(f"{dataset_path}/edges.csv")
            self.bus_map = pd.read_csv(f"{dataset_path}/bus_map.csv")
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到数据集文件！")

        self.num_timesteps = self.p_df.shape[0]
        self.num_buses = self.v_df.shape[1]

        # 2. 构建图结构 (使用导纳作为边权重)
        source = self.edges_df['from_bus'].values
        target = self.edges_df['to_bus'].values
        self.edge_index = torch.tensor([
            np.concatenate([source, target]),
            np.concatenate([target, source])
        ], dtype=torch.long)
        
        # 🔥 使用导纳模值作为边权重 (邻接矩阵权重)
        # 如果 edges.csv 包含导纳信息，则使用导纳作为权重
        if 'y_magnitude' in self.edges_df.columns:
            y_mag = self.edges_df['y_magnitude'].astype(float).values

            # --- 归一化导纳权重 (推荐) ---
            # 目的：避免少数“超短线路”产生极大导纳，导致消息传递被放大、训练不稳定。
            # 方案：p99 裁剪 + 均值归一化，使典型权重尺度约为 1。
            clip_value = np.percentile(y_mag, 99)
            y_mag_clipped = np.clip(y_mag, 0.0, clip_value)
            mean_value = float(np.mean(y_mag_clipped)) if y_mag_clipped.size > 0 else 1.0
            if mean_value <= 0:
                mean_value = 1.0
            y_mag_norm = y_mag_clipped / mean_value

            # 双向边，权重也要对称
            self.edge_weight = torch.tensor(
                np.concatenate([y_mag_norm, y_mag_norm]), dtype=torch.float
            )
            print(
                ">>> 使用导纳模值作为邻接矩阵权重 (已归一化: clip=p99, mean=1)，"
                f"边数: {len(self.edge_weight)}"
            )
            
            # 保存电阻、电抗、导纳信息供后续使用
            self.r_ohm = self.edges_df['r_ohm'].values if 'r_ohm' in self.edges_df.columns else None
            self.x_ohm = self.edges_df['x_ohm'].values if 'x_ohm' in self.edges_df.columns else None
            self.g_siemens = self.edges_df['g_siemens'].values if 'g_siemens' in self.edges_df.columns else None
            self.b_siemens = self.edges_df['b_siemens'].values if 'b_siemens' in self.edges_df.columns else None
        else:
            # 兼容旧数据：使用单位权重
            self.edge_weight = torch.ones(self.edge_index.shape[1], dtype=torch.float)
            print(">>> 警告: edges.csv 中无导纳数据，使用单位权重")

        # 3. 数据映射 (P, Q, V)
        # 先构建基础特征矩阵: [Time, Buses, 3]
        base_features = np.zeros((self.num_timesteps, self.num_buses, 3))
        self.labels = np.zeros((self.num_timesteps, self.num_buses))
        self.node_mask = np.zeros(self.num_buses, dtype=bool)

        # 填入 V
        base_features[:, :, 2] = self.v_df.values

        # 填入 P, Q
        load_to_bus = dict(zip(self.bus_map.index, self.bus_map['bus']))
        for load_idx in range(self.p_df.shape[1]):
            if load_idx in load_to_bus:
                bus_idx = load_to_bus[load_idx]
                self.node_mask[bus_idx] = True
                base_features[:, bus_idx, 0] = self.p_df.iloc[:, load_idx].values
                base_features[:, bus_idx, 1] = self.q_df.iloc[:, load_idx].values
                self.labels[:, bus_idx] = self.labels_df.iloc[:, load_idx].values

        # --- 🔥🔥 【核心修改】丰富的特征工程 🔥🔥 ---
        # 原始特征: P, Q, V (3维)
        # 新增特征:
        #   - S: 视在功率 = sqrt(P^2 + Q^2)
        #   - PF: 功率因数 = P / S (cos φ)
        #   - I: 电流估算 = S / V (标幺值下的近似)
        #   - Delta_P, Delta_Q, Delta_V: 时序差分
        #   - V_dev: 电压偏差 = |V - 1.0| (偏离标准电压的程度)
        #   - Q/P ratio: 无功有功比 (反映负荷特性)
        
        P = base_features[:, :, 0]  # [Time, Buses]
        Q = base_features[:, :, 1]
        V = base_features[:, :, 2]
        
        # 1. 视在功率 S = sqrt(P^2 + Q^2)
        S = np.sqrt(P**2 + Q**2)
        
        # 2. 功率因数 PF = P / S (避免除零)
        S_safe = np.where(S > 1e-6, S, 1e-6)
        PF = P / S_safe
        PF = np.clip(PF, -1.0, 1.0)  # 功率因数范围 [-1, 1]
        
        # 3. 电流估算 I = S / V (标幺值近似)
        V_safe = np.where(V > 0.01, V, 0.01)
        I_est = S / V_safe
        
        # 4. 电压偏差 V_dev = |V - 1.0|
        V_dev = np.abs(V - 1.0)
        
        # 5. 无功有功比 Q/P ratio (反映负荷特性，感性/容性)
        P_safe = np.where(np.abs(P) > 1e-6, P, 1e-6)
        QP_ratio = Q / P_safe
        QP_ratio = np.clip(QP_ratio, -10.0, 10.0)  # 限制范围避免极端值
        
        # 6. 时序差分 Delta
        delta_P = np.diff(P, axis=0, prepend=P[0:1, :])
        delta_Q = np.diff(Q, axis=0, prepend=Q[0:1, :])
        delta_V = np.diff(V, axis=0, prepend=V[0:1, :])
        
        # 7. 视在功率变化率
        delta_S = np.diff(S, axis=0, prepend=S[0:1, :])
        
        # === 组合所有特征 ===
        # 最终特征: [P, Q, V, S, PF, I_est, V_dev, QP_ratio, ΔP, ΔQ, ΔV, ΔS]
        # 共 12 维特征
        self.features = np.stack([
            P, Q, V,           # 原始特征 (3)
            S, PF, I_est,      # 派生物理量 (3)
            V_dev, QP_ratio,   # 状态指标 (2)
            delta_P, delta_Q, delta_V, delta_S  # 时序差分 (4)
        ], axis=2)
        
        self.feature_names = [
            'P (有功功率)', 'Q (无功功率)', 'V (电压)',
            'S (视在功率)', 'PF (功率因数)', 'I (电流估算)',
            'V_dev (电压偏差)', 'Q/P (无功有功比)',
            'ΔP', 'ΔQ', 'ΔV', 'ΔS'
        ]

        print(f">>> 特征工程完成！输入特征维度: {self.features.shape[2]}")
        print(f"    特征列表: {', '.join(self.feature_names)}")

        # 4. 归一化 (维度变了，Scaler也要自适应)
        scaler = StandardScaler()
        num_feats = self.features.shape[2]  # 现在是 6
        flat_features = self.features.reshape(-1, num_feats)
        flat_features = scaler.fit_transform(flat_features)
        self.features = flat_features.reshape(self.num_timesteps, self.num_buses, num_feats)

    # get_pyg_data_list 方法保持不变 (用于普通 GCN)
    def get_pyg_data_list(self):
        data_list = []
        for t in range(self.num_timesteps):
            x = torch.tensor(self.features[t], dtype=torch.float)
            y = torch.tensor(self.labels[t], dtype=torch.long)
            mask = torch.tensor(self.node_mask, dtype=torch.bool)
            data = Data(x=x, edge_index=self.edge_index, y=y, edge_attr=self.edge_weight)
            data.train_mask = mask
            data_list.append(data)
        return data_list

    # ============================================================
    # 🔥 新增: 用于时序GNN (TGCN) 的数据加载方法
    # ============================================================
    def get_temporal_data(self, seq_len=12):
        """
        生成时间窗口序列数据，用于时序GNN训练
        
        Args:
            seq_len: 时间窗口长度 (默认12，即用过去12个时间步预测当前)
        
        Returns:
            sequences: List[dict]，每个dict包含:
                - x_seq: [seq_len, num_nodes, num_features] 时间窗口特征
                - y: [num_nodes] 最后时刻的标签
                - edge_index: 图结构
                - edge_weight: 边权重(导纳)
                - node_mask: 有效节点掩码
        """
        sequences = []
        
        for t in range(seq_len, self.num_timesteps):
            # 取 [t-seq_len : t] 作为输入序列
            x_seq = torch.tensor(
                self.features[t - seq_len : t], dtype=torch.float
            )  # [seq_len, num_nodes, features]
            
            # 用最后时刻 (t-1) 的标签作为预测目标
            y = torch.tensor(self.labels[t - 1], dtype=torch.long)
            
            mask = torch.tensor(self.node_mask, dtype=torch.bool)
            
            sequences.append({
                'x_seq': x_seq,
                'y': y,
                'edge_index': self.edge_index,
                'edge_weight': self.edge_weight,
                'node_mask': mask
            })
        
        print(f">>> 时序数据生成完成: {len(sequences)} 个样本, 窗口长度={seq_len}")
        return sequences
    
    def get_temporal_tensors(self, seq_len=12):
        """
        返回完整的张量格式，适合批量训练
        
        Returns:
            X: [num_samples, seq_len, num_nodes, num_features]
            Y: [num_samples, num_nodes]
            edge_index, edge_weight, node_mask
        """
        num_samples = self.num_timesteps - seq_len
        
        X = np.zeros((num_samples, seq_len, self.num_buses, self.features.shape[2]))
        Y = np.zeros((num_samples, self.num_buses))
        
        for i in range(num_samples):
            t = i + seq_len
            X[i] = self.features[t - seq_len : t]
            Y[i] = self.labels[t - 1]
        
        X = torch.tensor(X, dtype=torch.float)
        Y = torch.tensor(Y, dtype=torch.long)
        node_mask = torch.tensor(self.node_mask, dtype=torch.bool)
        
        print(f">>> 时序张量生成完成: X={X.shape}, Y={Y.shape}")
        return X, Y, self.edge_index, self.edge_weight, node_mask