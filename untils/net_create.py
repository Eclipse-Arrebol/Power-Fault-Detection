import pandapower as pp
import numpy as np




def create_custom_lv_grid(n_feeders=5, nodes_per_feeder=30):
    print(f">>> 正在构建定制电网...")
    net = pp.create_empty_network()
    mv_bus = pp.create_bus(net, vn_kv=20, name="MV Bus", geodata=(0, 0))
    pp.create_ext_grid(net, bus=mv_bus, vm_pu=1.0)
    lv_trafo_bus = pp.create_bus(net, vn_kv=0.4, name="LV Trafo Bus", geodata=(0, 0))
    pp.create_transformer(net, hv_bus=mv_bus, lv_bus=lv_trafo_bus, std_type="0.4 MVA 20/0.4 kV")

    load_count = 0
    for i in range(n_feeders):
        previous_bus = lv_trafo_bus
        for j in range(nodes_per_feeder):
            load_count += 1
            angle = (2 * np.pi / n_feeders) * i
            distance = (j + 1) * 2.0
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            new_bus = pp.create_bus(net, vn_kv=0.4, name=f"Line{i}_Node{j}", geodata=(x, y))
            pp.create_line_from_parameters(net, from_bus=previous_bus, to_bus=new_bus, length_km=0.05,
                                           r_ohm_per_km=0.206, x_ohm_per_km=0.080, c_nf_per_km=261.0, max_i_ka=0.270,
                                           name=f"Line_{i}_{j}")
            pp.create_load(net, bus=new_bus, p_mw=0.0, q_mvar=0.0, name=f"Load_{load_count}")
            previous_bus = new_bus
    return net