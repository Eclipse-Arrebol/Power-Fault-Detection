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


from untils.net_create import create_custom_lv_grid
from untils.get_data import extract_lv_simbench_data
from untils.fate_inject import inject_anomalies



# ==========================================
# 💾 新增：数据保存函数
# ==========================================
def save_dataset_for_gnn(net, p_df, q_df, v_df, labels_df, folder="dataset"):
    """
    将所有训练所需的数据保存到 CSV 文件
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    print(f">>> [6/6] 正在保存数据集到 '{folder}/' 文件夹...")

    # 1. 保存时间序列特征 (P, Q, V, Labels)
    # index=False 代表不保存时间戳索引，只保存纯数据矩阵
    p_df.to_csv(f"{folder}/p_mw.csv", index=False)
    q_df.to_csv(f"{folder}/q_mvar.csv", index=False)
    v_df.to_csv(f"{folder}/vm_pu.csv", index=False)
    labels_df.to_csv(f"{folder}/labels.csv", index=False)

    # 2. 保存图结构 (边列表 Edge List)
    # GNN 需要知道哪些节点是相连的
    # net.line 里的 from_bus 和 to_bus 就是图的边
    edges = net.line[['from_bus', 'to_bus', 'length_km', 'r_ohm_per_km', 'x_ohm_per_km']]
    edges.to_csv(f"{folder}/edges.csv", index=False)

    # 3. 保存节点映射关系
    # 因为 P/Q/Label 是按 Load 排列的，但图结构是按 Bus 排列的
    # 我们需要一个表来查 "Load_0" 挂在哪个 "Bus" 上
    # 这样 GNN 才能把负荷数据映射到图节点上
    bus_map = net.load[['name', 'bus']]
    bus_map.to_csv(f"{folder}/bus_map.csv", index=False)

    print("    - 保存成功！你可以直接用这些 CSV 训练模型了。")


# ==========================================
# 主程序
# ==========================================
def run_simulation_with_anomalies():
    # 1. 生成电网
    net = create_custom_lv_grid(n_feeders=5, nodes_per_feeder=30)

    # 2. 提取干净数据
    n_steps = 96
    clean_p, clean_q = extract_lv_simbench_data(len(net.load), n_steps)
    clean_p.columns = net.load.index
    clean_q.columns = net.load.index

    # 3. 注入异常
    p_data, q_data, labels = inject_anomalies(clean_p, clean_q, anomaly_ratio=0.02)

    # 4. 绑定控制器
    ds_p = DFData(p_data)
    ds_q = DFData(q_data)
    print(">>> [4/5] 绑定控制器...")
    ConstControl(net, element='load', variable='p_mw', element_index=net.load.index,
                 data_source=ds_p, profile_name=net.load.index)
    ConstControl(net, element='load', variable='q_mvar', element_index=net.load.index,
                 data_source=ds_q, profile_name=net.load.index)

    # 5. 运行仿真
    print(">>> [5/5] 启动仿真...")
    output_path = "./results"
    ow = ts.OutputWriter(net, output_path=output_path, output_file_type=".json")
    # 记录所有节点的电压
    ow.log_variable('res_bus', 'vm_pu')

    try:
        ts.run_timeseries(net, time_steps=range(n_steps), algorithm="nr")
        print("\n✅✅✅ 仿真成功！")

        # --- 提取仿真结果中的电压数据 ---
        # 结果在 ow.output['res_bus.vm_pu'] 中
        vm_results = ow.output['res_bus.vm_pu']

        # --- 6. 保存数据 (新增步骤) ---
        save_dataset_for_gnn(net, p_data, q_data, vm_results, labels)

        # --- 7. 画图 (仅做展示) ---
        anomalous_cols = labels.columns[labels.sum() > 0]
        if len(anomalous_cols) > 0:
            target_col = anomalous_cols[0]

            print(f"\n正在绘制用户 [{target_col}] 的对比图...")
            fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            ax[0].plot(clean_p[target_col], 'g--', label="Normal")
            ax[0].plot(p_data[target_col], 'r-', label="Injected")
            ax[0].set_ylabel("Active Power [MW]")
            ax[0].set_title(f"User {target_col} Load Profile")
            ax[0].legend()

            ax[1].plot(labels[target_col], 'k-', drawstyle='steps-post')
            ax[1].set_ylabel("Label")
            ax[1].set_ylim(-0.5, 3.5)
            ax[1].grid(True)
            plt.show()

    except Exception as e:
        print(f"❌ 失败: {e}")


if __name__ == "__main__":
    run_simulation_with_anomalies()