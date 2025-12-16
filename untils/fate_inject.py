import simbench as sb
import pandapower as pp
import pandapower.timeseries as ts
from pandapower.control import ConstControl
from pandapower.timeseries.data_sources.frame_data import DFData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandapower.plotting as pplot
import random
import os  # 新增：用于创建文件夹



def inject_anomalies(df_p, df_q, anomaly_ratio=0.05):
    print(f">>> [3/5] 正在注入异常 (比例: {anomaly_ratio * 100}%)...")
    dirty_p = df_p.copy()
    dirty_q = df_q.copy()
    labels = pd.DataFrame(0, index=df_p.index, columns=df_p.columns)

    n_rows, n_cols = df_p.shape
    n_anomalies = int(n_rows * n_cols * anomaly_ratio)

    for _ in range(n_anomalies):
        row_idx = np.random.randint(0, n_rows)
        col_idx = np.random.randint(0, n_cols)

        anomaly_type = np.random.choice([1, 2, 3])
        duration = np.random.randint(1, 5)
        end_idx = min(row_idx + duration, n_rows)

        if anomaly_type == 1:  # 突增
            factor = np.random.uniform(2.0, 5.0)
            dirty_p.iloc[row_idx:end_idx, col_idx] *= factor
            labels.iloc[row_idx:end_idx, col_idx] = 1
        elif anomaly_type == 2:  # 突降
            factor = np.random.uniform(0.0, 0.1)
            dirty_p.iloc[row_idx:end_idx, col_idx] *= factor
            labels.iloc[row_idx:end_idx, col_idx] = 2
        elif anomaly_type == 3:  # 无功干扰
            dirty_q.iloc[row_idx:end_idx, col_idx] += 0.005
            labels.iloc[row_idx:end_idx, col_idx] = 3

    print(f"    - 注入完成！共注入约 {n_anomalies} 个异常事件。")
    return dirty_p, dirty_q, labels