import pandapower as pp
import pandapower.plotting as pplot
import simbench as sb
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# 忽略 pandas 的 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


# ==========================================
# 🏗️ 核心函数：克隆指定拓扑
# ==========================================
def create_cloned_simbench_grid(topo_code="1-LV-rural3--2-no_sw"):
    """
    完全克隆 SimBench 的拓扑结构（物理参数），但重建为纯净的 pandapower 网络。
    并自动生成绘图坐标。
    """
    print(f">>> [Net] 正在克隆目标拓扑: {topo_code} ...")

    # 1. 获取原始数据 (只为了读取参数)
    try:
        raw_net = sb.get_simbench_net(topo_code)
    except Exception as e:
        print(f"❌ 错误: 无法加载 SimBench 拓扑。请确保已安装 simbench 库。\n{e}")
        return None

    # 2. 初始化纯净网络
    net = pp.create_empty_network()

    # --- 映射表：旧 Bus ID -> 新 Bus ID ---
    # 防止原始索引不连续导致错乱
    old_to_new_bus = {}

    # 3. 复制 Bus (节点)
    # 注意：我们先暂时忽略原始坐标，最后统一生成，防止报错
    for old_idx, row in raw_net.bus.iterrows():
        new_idx = pp.create_bus(
            net,
            vn_kv=row['vn_kv'],
            name=f"Bus_{old_idx}"
        )
        old_to_new_bus[old_idx] = new_idx

    # 4. 复制 External Grid (外部电源)
    for _, row in raw_net.ext_grid.iterrows():
        pp.create_ext_grid(
            net,
            bus=old_to_new_bus[row['bus']],
            vm_pu=1.02,
            name="External Grid"
        )

    # 5. 复制 Transformer (变压器)
    for _, row in raw_net.trafo.iterrows():
        pp.create_transformer_from_parameters(
            net,
            hv_bus=old_to_new_bus[row['hv_bus']],
            lv_bus=old_to_new_bus[row['lv_bus']],
            sn_mva=row['sn_mva'],
            vn_hv_kv=row['vn_hv_kv'],
            vn_lv_kv=row['vn_lv_kv'],
            vkr_percent=row['vkr_percent'],
            vk_percent=row['vk_percent'],
            pfe_kw=row['pfe_kw'],
            i0_percent=row['i0_percent'],
            name=row['name']
        )

    # 6. 复制 Line (线路) - 核心物理参数
    for _, row in raw_net.line.iterrows():
        pp.create_line_from_parameters(
            net,
            from_bus=old_to_new_bus[row['from_bus']],
            to_bus=old_to_new_bus[row['to_bus']],
            length_km=row['length_km'],
            r_ohm_per_km=row['r_ohm_per_km'],
            x_ohm_per_km=row['x_ohm_per_km'],
            c_nf_per_km=row['c_nf_per_km'],
            max_i_ka=row['max_i_ka'],
            name=row['name']
        )

    # 7. 重建 Load (负荷)
    # 在原版有负荷的地方挂载空负荷
    for _, row in raw_net.load.iterrows():
        pp.create_load(
            net,
            bus=old_to_new_bus[row['bus']],
            p_mw=0.0,  # 初始设为0
            q_mvar=0.0,
            name=f"Load_at_{row['bus']}"
        )

    print(f"    - 克隆完成！包含: {len(net.bus)} 节点, {len(net.line)} 线路, {len(net.load)} 负荷")

    # ==========================================
    # 🔥 关键修复：自动生成绘图坐标
    # ==========================================
    # SimBench 自带的 geodata 经常损坏或缺失，导致 AttributeError。
    # 这里我们使用 pandapower 的图算法自动计算拓扑坐标。
    print("    - 正在自动计算拓扑布局坐标 (Generic Coordinates)...")
    try:
        pplot.create_generic_coordinates(net, overwrite=True)
    except Exception as e:
        print(f"    ⚠️ 警告: 坐标生成失败，画图可能会重叠。原因: {e}")

    return net


# ==========================================
# 🎨 绘图展示脚本
# ==========================================
def visualize_network():
    # 1. 调用克隆函数
    target_code = "1-LV-rural3--2-no_sw"
    net = create_cloned_simbench_grid(target_code)

    if net is None:
        return

    print("\n>>> 正在绘制拓扑图...")

    # 创建图表对象
    fig, ax = plt.subplots(figsize=(12, 8))

    # 2. 绘图
    # 因为我们用了 create_generic_coordinates，这里一定能画出来
    pplot.simple_plot(
        net,
        plot_loads=True,  # 显示负载
        plot_sgens=False,
        show_plot=False,
        ax=ax,
        bus_size=0.7,  # 节点稍微画小一点，因为节点多
        line_width=1.0
    )

    # 3. 添加装饰
    plt.title(f"Cloned Topology: {target_code}", fontsize=15)
    plt.xlabel("Generic X Coordinate", fontsize=12)
    plt.ylabel("Generic Y Coordinate", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)

    # 4. 显示
    plt.tight_layout()
    plt.show()
    print("✅ 绘图完成！此图展示了 SimBench 目标拓扑的物理结构。")


if __name__ == "__main__":
    visualize_network()