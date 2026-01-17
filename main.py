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


from untils.net_create import create_cloned_simbench_grid
from untils.get_data import extract_lv_simbench_data
from untils.fate_inject import inject_anomalies



# ==========================================
# 💾 新增：数据保存函数
# ==========================================
def save_dataset_for_gnn(net, p_df, q_df, v_df, labels_df, i_bus_df=None, folder="dataset"):
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
    
    # 🔥 新增：保存母线电流数据（如果有的话）
    if i_bus_df is not None:
        i_bus_df.to_csv(f"{folder}/i_bus_ka.csv", index=False)
        print(f"    - 已保存母线电流数据到 i_bus_ka.csv (基于母线负荷聚合计算)")
    
    # 🔥 计算并保存负荷电流数据
    # 对于每个负荷，根据其挂载的母线电压计算电流
    # I = S / (√3 * V)
    print("    - 正在计算负荷电流数据...")
    
    # 获取负荷到母线的映射
    load_to_bus = dict(zip(net.load.index, net.load.bus))
    
    # 获取母线的基准电压 (kV)
    bus_vn_kv = net.bus.vn_kv.to_dict()
    
    # 初始化电流矩阵（与P、Q相同维度）
    i_load_df = pd.DataFrame(index=p_df.index, columns=p_df.columns, dtype=float)
    
    for load_idx in p_df.columns:
        if load_idx in load_to_bus:
            bus_idx = load_to_bus[load_idx]
            
            # 获取该母线的电压时间序列和基准电压
            if bus_idx in v_df.columns and bus_idx in bus_vn_kv:
                v_pu = v_df[bus_idx].values  # 母线电压 [p.u.]
                v_base_kv = bus_vn_kv[bus_idx]  # 基准电压 [kV]
                v_kv = v_pu * v_base_kv  # 实际电压 [kV]
                
                p_load = p_df[load_idx].values  # 有功功率 [MW]
                q_load = q_df[load_idx].values  # 无功功率 [Mvar]
                
                # 计算视在功率 S = √(P² + Q²) [MVA]
                s_load = np.sqrt(p_load**2 + q_load**2)
                
                # 计算电流 I = S / (√3 * V) [kA]
                # 对于三相系统: I = S / (√3 * V_line)
                # 避免除零
                v_kv_safe = np.where(v_kv > 0.01, v_kv, 0.01)
                i_load = s_load / (np.sqrt(3) * v_kv_safe)  # [kA]
                
                i_load_df[load_idx] = i_load
            else:
                # 如果找不到对应母线电压，设为0
                i_load_df[load_idx] = 0.0
        else:
            i_load_df[load_idx] = 0.0
    
    # 保存负荷电流数据
    i_load_df.to_csv(f"{folder}/i_load_ka.csv", index=False)
    print(f"    - 已计算并保存负荷电流数据到 i_load_ka.csv (基于 I = S / (√3 * V))")

    # 2. 保存图结构 (边列表 Edge List)
    # GNN 需要知道哪些节点是相连的
    # net.line 里的 from_bus 和 to_bus 就是图的边
    edges = net.line[['from_bus', 'to_bus', 'length_km', 'r_ohm_per_km', 'x_ohm_per_km']].copy()
    
    # 计算实际电阻和电抗 (R = r_ohm_per_km * length_km, X = x_ohm_per_km * length_km)
    edges['r_ohm'] = edges['r_ohm_per_km'] * edges['length_km']
    edges['x_ohm'] = edges['x_ohm_per_km'] * edges['length_km']
    
    # 计算导纳 Y = 1 / Z = 1 / (R + jX)
    # 导纳模值 |Y| = 1 / |Z| = 1 / sqrt(R^2 + X^2)
    # 导纳实部 G = R / (R^2 + X^2)
    # 导纳虚部 B = -X / (R^2 + X^2)
    z_squared = edges['r_ohm']**2 + edges['x_ohm']**2
    z_squared = z_squared.replace(0, 1e-10)  # 避免除零
    edges['g_siemens'] = edges['r_ohm'] / z_squared  # 电导 (导纳实部)
    edges['b_siemens'] = -edges['x_ohm'] / z_squared  # 电纳 (导纳虚部)
    edges['y_magnitude'] = 1 / np.sqrt(z_squared)     # 导纳模值
    
    edges.to_csv(f"{folder}/edges.csv", index=False)
    print(f"    - 已计算并保存电阻(R)、电抗(X)和导纳(Y)到 edges.csv")

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
    net = create_cloned_simbench_grid()

    # 2. 提取干净数据
    n_steps = 6720
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
    # 🔥 记录母线电流（从每个母线流出的总电流）
    # 注意：pandapower的res_bus可能没有直接的电流字段，需要检查是否有p_mw和q_mvar
    # 我们可以通过母线的功率和电压计算母线电流 I = S / (√3 * V)

    try:
        ts.run_timeseries(net, time_steps=range(n_steps), algorithm="nr")
        print("\n✅✅✅ 仿真成功！")

        # --- 提取仿真结果中的电压数据 ---
        # 结果在 ow.output['res_bus.vm_pu'] 中
        vm_results = ow.output['res_bus.vm_pu']
        
        # --- 计算母线电流数据 ---
        # 方法：通过母线负荷聚合计算母线电流
        print(">>> [6/7] 正在计算母线电流...")
        
        # 通过负荷-母线映射反向聚合计算母线电流
        # 对于每个母线，汇总其上所有负荷的功率，然后计算电流
        bus_vn_kv = net.bus.vn_kv.to_dict()
        load_to_bus = dict(zip(net.load.index, net.load.bus))
        
        # 初始化母线电流矩阵 (时间步 x 母线数)
        i_bus_df = pd.DataFrame(0.0, index=vm_results.index, columns=vm_results.columns, dtype=float)
        
        # 对每个母线，汇总其上的负荷功率
        bus_p = pd.DataFrame(0.0, index=p_data.index, columns=vm_results.columns)
        bus_q = pd.DataFrame(0.0, index=q_data.index, columns=vm_results.columns)
        
        for load_idx in net.load.index:
            bus_idx = load_to_bus[load_idx]
            if bus_idx in bus_p.columns:
                bus_p[bus_idx] += p_data[load_idx]
                bus_q[bus_idx] += q_data[load_idx]
        
        # 计算每个母线的电流 I = S / (√3 * V)
        for bus_idx in vm_results.columns:
            if bus_idx in bus_vn_kv:
                v_pu = vm_results[bus_idx].values  # 母线电压 [p.u.]
                v_base_kv = bus_vn_kv[bus_idx]  # 基准电压 [kV]
                v_kv = v_pu * v_base_kv  # 实际电压 [kV]
                
                p_bus = bus_p[bus_idx].values  # 母线总有功 [MW]
                q_bus = bus_q[bus_idx].values  # 母线总无功 [Mvar]
                
                # 计算视在功率 S = √(P² + Q²) [MVA]
                s_bus = np.sqrt(p_bus**2 + q_bus**2)
                
                # 计算电流 I = S / (√3 * V) [kA]
                v_kv_safe = np.where(v_kv > 0.01, v_kv, 0.01)
                i_bus_df[bus_idx] = s_bus / (np.sqrt(3) * v_kv_safe)
        
        print(f"    - 已计算母线电流数据 (基于母线负荷聚合和公式 I = S / (√3 * V))")

        # --- 7. 保存数据 (新增步骤) ---
        save_dataset_for_gnn(net, p_data, q_data, vm_results, labels, i_bus_df)

        # --- 8. 画图 (仅做展示) ---
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