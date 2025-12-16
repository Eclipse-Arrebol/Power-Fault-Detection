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



def extract_lv_simbench_data(target_n_loads, n_steps=96):
    source_net = sb.get_simbench_net("1-MVLV-urban-all-0-sw")
    load_bus_voltages = source_net.load.bus.map(source_net.bus.vn_kv)
    lv_load_indices = source_net.load[load_bus_voltages <= 1.0].index
    profiles = sb.get_absolute_values(source_net, profiles_instead_of_study_cases=True)
    p_data = profiles[("load", "p_mw")][lv_load_indices].iloc[:n_steps, :target_n_loads]
    q_data = profiles[("load", "q_mvar")][lv_load_indices].iloc[:n_steps, :target_n_loads]
    return p_data, q_data