import torch
import pandas as pd
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from src.dataset import PowerGridDataset
from src.models import GCN

# 定义异常类型的文字描述
CLASS_NAMES = {
    0: "正常 (Normal)",
    1: "负荷突增 (Overload)",
    2: "负荷丢失 (Drop)",
    3: "无功干扰 (High Q)"
}


def predict_anomalies():
    # 1. 加载数据
    print(">>> [1/3] 加载数据...")
    dataset = PowerGridDataset(dataset_path="dataset")
    data_list = dataset.get_pyg_data_list()

    # 【修改点1】将 shuffle 改为 True，随机打乱数据，这样能看到各种类型
    _, test_data = train_test_split(data_list, test_size=0.2, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # 2. 加载模型
    print(">>> [2/3] 加载模型权重...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features=3, num_classes=4).to(device)
    model.load_state_dict(torch.load("gcn_model.pth"))
    model.eval()

    # 3. 开始诊断
    print(">>> [3/3] 开始随机抽检...")
    print(f"{'样本序号':<10} | {'节点ID':<10} | {'预测类型':<20} | {'真实标签':<20} | {'状态'}")
    print("-" * 80)

    # 【修改点2】添加统计计数器，看看测试集里到底有什么
    stats = {0: 0, 1: 0, 2: 0, 3: 0}
    correct_stats = {0: 0, 1: 0, 2: 0, 3: 0}

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)

            # 统计真实标签分布
            for cls in range(4):
                # 统计当前 batch 里有多少个该类别的节点 (只看 mask=True)
                mask = batch.train_mask
                count = (batch.y[mask] == cls).sum().item()
                stats[cls] += count

                # 统计预测对的数量
                correct = ((pred[mask] == batch.y[mask]) & (batch.y[mask] == cls)).sum().item()
                correct_stats[cls] += correct

            # 打印异常详情 (为了防止刷屏，只打印前 20 个检测到的异常)
            for node_idx in range(batch.num_nodes):
                if batch.train_mask[node_idx]:
                    p = pred[node_idx].item()
                    t = batch.y[node_idx].item()

                    # 只要预测或者真实是异常，就打印出来看看
                    if (p != 0 or t != 0) and i < 20:
                        status = "✅ 抓到了" if p == t else f"❌ 错判 (预:{p} 真:{t})"
                        print(
                            f"Sample={i:<4} | Node {node_idx:<5} | {CLASS_NAMES[p]:<18} | {CLASS_NAMES[t]:<18} | {status}")

    print("-" * 80)
    print(">>> 测试集全部分布统计 (真实标签数量):")
    print(f"  正常样本 (Class 0): {stats[0]}")
    print(f"  负荷突增 (Class 1): {stats[1]} (模型抓获: {correct_stats[1]})")
    print(f"  负荷丢失 (Class 2): {stats[2]} (模型抓获: {correct_stats[2]})")
    print(f"  无功干扰 (Class 3): {stats[3]} (模型抓获: {correct_stats[3]})")

if __name__ == "__main__":
    predict_anomalies()