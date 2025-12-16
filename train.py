import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 导入我们在 src 里写的模块
from src.dataset import PowerGridDataset
from src.models import GCN


def train_main():
    # 1. 准备数据
    print(">>> [1/4] 加载数据集...")
    dataset = PowerGridDataset(dataset_path="dataset")
    data_list = dataset.get_pyg_data_list()

    # 2. 划分训练集/测试集
    train_data, test_data = train_test_split(data_list, test_size=0.2, shuffle=False)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    # 3. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">>> [2/4] 初始化 GCN 模型 (Device: {device})...")

    # 输入特征=3 (P,Q,V)，输出类别=4
    model = GCN(num_features=3, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # 4. 开始训练循环
    print(">>> [3/4] 开始训练...")
    loss_history = []

    model.train()
    for epoch in range(100):  # 训练 100 轮
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)

            # 关键：只计算 mask=True 的节点的 Loss
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:03d} | Loss: {avg_loss:.4f}")

    # 5. 评估与保存
    print(">>> [4/4] 评估模型...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch).argmax(dim=1)

            # 只统计有效节点
            mask = batch.train_mask
            correct += (pred[mask] == batch.y[mask]).sum().item()
            total += mask.sum().item()

    acc = correct / total
    print("=" * 40)
    print(f"✅ 最终测试集准确率: {acc * 100:.2f}%")
    print("=" * 40)

    # 保存模型
    torch.save(model.state_dict(), "gcn_model.pth")

    # 画图
    plt.plot(loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    train_main()