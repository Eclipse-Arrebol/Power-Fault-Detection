import numpy as np
import pandas as pd


def inject_anomalies(df_p, df_q, anomaly_ratio=0.05):
    """
    向负荷数据中注入异常 (增强版：特征更显著)
    :param df_p: 有功功率 DataFrame
    :param df_q: 无功功率 DataFrame
    :param anomaly_ratio: 异常比例
    :return: dirty_p, dirty_q, labels
    """
    print(f">>> [3/5] 正在注入异常 (比例: {anomaly_ratio * 100}%)...")

    # 复制数据
    dirty_p = df_p.copy()
    dirty_q = df_q.copy()

    # 初始化标签矩阵
    labels = pd.DataFrame(0, index=df_p.index, columns=df_p.columns)

    n_rows, n_cols = df_p.shape
    n_anomalies = int(n_rows * n_cols * anomaly_ratio)

    # --- 【关键改进1】计算每个用户的历史最大值 ---
    # 用作判断 "突增" 是否显著的基准线
    max_p_profile = df_p.max(axis=0)

    for _ in range(n_anomalies):
        # 1. 随机选位置
        row_idx = np.random.randint(0, n_rows)
        col_idx = np.random.randint(0, n_cols)

        anomaly_type = np.random.choice([1, 2, 3])
        duration = np.random.randint(1, 5)
        end_idx = min(row_idx + duration, n_rows)

        # 2. 注入逻辑
        if anomaly_type == 1:  # 负荷突增 (Overload)
            # --- 【改进】强制突破历史峰值 ---
            # 仅仅乘以 2~5 倍可能还不够大（比如深夜基数很小）
            # 我们取 max(原值*随机倍数, 历史峰值*1.5)
            # 确保这个异常值比平时最大的时候还要大，这就很明显了

            # 基础倍数
            random_factor = np.random.uniform(2.0, 5.0)
            multiplied_val = dirty_p.iloc[row_idx:end_idx, col_idx] * random_factor

            # 阈值基线 (历史最大值的 1.5 倍)
            threshold_val = max_p_profile[col_idx] * 1.5

            # 取两者中更大的那个
            dirty_p.iloc[row_idx:end_idx, col_idx] = np.maximum(multiplied_val, threshold_val)
            labels.iloc[row_idx:end_idx, col_idx] = 1

        elif anomaly_type == 2:  # 负荷丢失 (Drop)
            # --- 【改进】只在有负荷时才注入 ---
            # 如果原本就是 0 (比如深夜)，注入丢失也没有意义，模型学不到
            current_val = dirty_p.iloc[row_idx, col_idx]

            if current_val > 0.0005:  # 只有当原本负荷 > 0.5kW 时才搞破坏
                factor = np.random.uniform(0.0, 0.1)  # 变为原来的 0~10%
                dirty_p.iloc[row_idx:end_idx, col_idx] *= factor
                labels.iloc[row_idx:end_idx, col_idx] = 2
            else:
                # 如果原本就是0，这次机会作废，不做任何标记
                pass

        elif anomaly_type == 3:  # 无功干扰 (High Q)
            # 这个特征本来就很强，保持不变
            dirty_q.iloc[row_idx:end_idx, col_idx] += 0.005
            labels.iloc[row_idx:end_idx, col_idx] = 3

    print(f"    - 注入完成！共尝试注入 {n_anomalies} 个异常事件。")
    return dirty_p, dirty_q, labels