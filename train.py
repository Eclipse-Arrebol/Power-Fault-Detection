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

    # 【修改点 1】：学习率从 0.01 改为 0.005，因为加权后 Loss 会变大，步长小一点更稳
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # 【修改点 2】：定义暴力权重 (Violent Weights)
    # 告诉模型：猜错一个 Class 2 (丢失)，相当于猜错 300 个正常样本！
    # 顺序对应: [Class 0, Class 1, Class 2, Class 3]
    class_weights = torch.tensor([1.0, 50.0, 300.0, 20.0]).to(device)

    # 4. 开始训练循环
    print(">>> [3/4] 开始训练...")
    loss_history = []

    model.train()
    for epoch in range(300):  # 训练 300 轮
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)

            # 【修改点 3】：把 class_weights 传进去
            # 关键：只计算 mask=True 的节点的 Loss，并应用权重
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask], weight=class_weights)

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

    # 新增：顺便统计一下各类的准确率，防止只看总分
    class_correct = [0, 0, 0, 0]
    class_total = [0, 0, 0, 0]

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch).argmax(dim=1)

            # 只统计有效节点
            mask = batch.train_mask

            # 总准确率
            correct += (pred[mask] == batch.y[mask]).sum().item()
            total += mask.sum().item()

            # 分类统计 (可选，方便调试)
            for c in range(4):
                # 找出 mask=True 且 label=c 的节点
                c_mask = mask & (batch.y == c)
                class_total[c] += c_mask.sum().item()
                class_correct[c] += (pred[c_mask] == batch.y[c_mask]).sum().item()

    acc = correct / total
    print("=" * 40)
    print(f"✅ 最终测试集总准确率: {acc * 100:.2f}%")
    print(f"   - Class 0 (正常): {class_correct[0]}/{class_total[0]}")
    print(f"   - Class 1 (突增): {class_correct[1]}/{class_total[1]}")
    print(f"   - Class 2 (丢失): {class_correct[2]}/{class_total[2]}")
    print(f"   - Class 3 (无功): {class_correct[3]}/{class_total[3]}")
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