import pandapower as pp
import numpy as np
import random



def create_custom_lv_grid(n_feeders=5, min_nodes=15, max_nodes=45):
    """
        生成一个【高度拟真】的随机辐射状低压台区
        :param n_feeders: 馈线数量 (主干线)
        :param min_nodes: 每条线最少多少个用户
        :param max_nodes: 每条线最多多少个用户
        """
    print(f">>> [Net] 正在构建随机异构电网 (Feeders={n_feeders}, Nodes={min_nodes}~{max_nodes})...")

    # 1. 初始化空网络
    net = pp.create_empty_network()

    # 2. 创建源头：中压母线 (MV) 和 变压器 (Trafo)
    # 坐标设为原点 (0,0)
    mv_bus = pp.create_bus(net, vn_kv=20, name="MV Bus", geodata=(0, 0))
    pp.create_ext_grid(net, bus=mv_bus, vm_pu=1.0)

    lv_trafo_bus = pp.create_bus(net, vn_kv=0.4, name="LV Trafo Bus", geodata=(0, 0))
    pp.create_transformer(net, hv_bus=mv_bus, lv_bus=lv_trafo_bus, std_type="0.4 MVA 20/0.4 kV")

    # 3. 随机生成各条馈线
    load_count = 0
    total_lines = 0

    # 预先生成每条线的节点数，打破对称性
    # 比如: [20, 42, 15, 33, 28]
    nodes_per_feeder_list = [random.randint(min_nodes, max_nodes) for _ in range(n_feeders)]

    for i, n_nodes in enumerate(nodes_per_feeder_list):
        previous_bus = lv_trafo_bus

        # 设定这条线的“主攻方向” (基础角度)
        base_angle = (2 * np.pi / n_feeders) * i

        # 记录当前的物理距离 (用于画图坐标推算)
        current_x = 0.0
        current_y = 0.0

        for j in range(n_nodes):
            load_count += 1
            total_lines += 1

            # --- A. 随机物理参数 (核心) ---
            # 线路长度随机化：0.03km 到 0.08km 之间 (30米~80米一根杆)
            # 这直接决定了线路阻抗 R 和 X 的大小！
            line_len = random.uniform(0.03, 0.08)

            # --- B. 随机几何参数 (为了画图好看) ---
            # 在主方向上加一点随机抖动 (-15度 到 +15度)
            angle_jitter = random.uniform(-0.25, 0.25)
            actual_angle = base_angle + angle_jitter

            # 计算新坐标 (累加)
            # 这里的 *20 只是为了画图时拉开距离，不影响物理计算
            step_distance = line_len * 20
            current_x += step_distance * np.cos(actual_angle)
            current_y += step_distance * np.sin(actual_angle)

            # --- C. 创建组件 ---
            # 1. 创建节点
            new_bus = pp.create_bus(net, vn_kv=0.4, name=f"F{i}_Node{j}", geodata=(current_x, current_y))

            # 2. 创建线路
            # 使用 create_line_from_parameters 保证参数完全受控
            pp.create_line_from_parameters(
                net,
                from_bus=previous_bus,
                to_bus=new_bus,
                length_km=line_len,  # <--- 这里用了随机长度
                r_ohm_per_km=0.206,  # 典型铝芯线电阻
                x_ohm_per_km=0.080,  # 典型电抗
                c_nf_per_km=261.0,
                max_i_ka=0.270,
                name=f"Line_F{i}_{j}"
            )

            # 3. 创建负载 (占位)
            pp.create_load(net, bus=new_bus, p_mw=0.0, q_mvar=0.0, name=f"Load_{load_count}")

            # 迭代指针
            previous_bus = new_bus

    print(f"    - 构建完成！")
    print(f"    - 拓扑结构: {nodes_per_feeder_list}")
    print(f"    - 总负载数: {len(net.load)}")
    print(f"    - 总节点数: {len(net.bus)}")

    return net