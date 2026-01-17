"""script/show_bus_load_mapping.py

用途：查看功率表(p_mw/q_mvar)的列(负荷点)如何映射到电压表(vm_pu)的母线列。

示例：
  python script/show_bus_load_mapping.py --bus 77
  python script/show_bus_load_mapping.py --export result/bus_load_mapping.csv
"""

import argparse
import os

import pandas as pd


def load_mapping(dataset_dir: str = "dataset") -> pd.DataFrame:
    bus_map_path = os.path.join(dataset_dir, "bus_map.csv")
    bus_map = pd.read_csv(bus_map_path)

    # bus_map 的行序号就是负荷列索引 load_idx (与 p_mw/q_mvar 的列一一对应)
    bus_map = bus_map.reset_index(drop=False).rename(columns={"index": "load_idx"})
    # 统一字段名
    if "bus" not in bus_map.columns:
        raise ValueError("bus_map.csv 缺少 'bus' 列")
    if "name" not in bus_map.columns:
        bus_map["name"] = ""

    bus_map["bus"] = bus_map["bus"].astype(int)
    bus_map["load_idx"] = bus_map["load_idx"].astype(int)
    return bus_map[["load_idx", "name", "bus"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset", help="数据集目录，默认 dataset")
    parser.add_argument("--bus", type=int, default=None, help="母线号(bus id)，例如 77")
    parser.add_argument("--export", default=None, help="导出映射CSV路径，例如 result/bus_load_mapping.csv")
    args = parser.parse_args()

    mapping = load_mapping(args.dataset)

    print("=" * 70)
    print("负荷列(load_idx) → 母线列(bus) 对应关系")
    print("说明：p_mw/q_mvar 的第 load_idx 列，对应 vm_pu 的第 bus 列")
    print("=" * 70)

    if args.bus is not None:
        rows = mapping[mapping["bus"] == args.bus].sort_values("load_idx")
        if rows.empty:
            print(f"bus={args.bus} 没有负荷映射（该母线可能无负荷，仅有电压）")
        else:
            print(f"bus={args.bus} 对应的负荷列：")
            for _, r in rows.iterrows():
                print(f"  load_idx={r['load_idx']:>3}  name={r['name']}  -> vm_pu列(bus)={r['bus']}")

    # 简单统计：每个母线挂了几个负荷
    counts = mapping["bus"].value_counts().sort_index()
    print("\n[统计] 母线挂载负荷数（前20个非零母线）")
    print(counts.head(20).to_string())

    if args.export:
        os.makedirs(os.path.dirname(args.export), exist_ok=True)
        mapping.to_csv(args.export, index=False, encoding="utf-8-sig")
        print(f"\n已导出映射表到: {args.export}")


if __name__ == "__main__":
    main()
