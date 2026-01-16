"""
因果关系图绘制与分析模块
包含格兰杰因果图的可视化和统计分析功能
"""
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_causal_graph(causal_matrix, threshold=0.1, save_path="img/causal_graph.png", 
                      node_labels=None, figsize=(14, 10)):
    """
    绘制因果关系图
    
    Args:
        causal_matrix: [num_nodes, num_nodes] 因果权重矩阵
        threshold: 边权重阈值，低于此值的边不显示
        save_path: 保存路径
        node_labels: 节点标签字典 {idx: label}
        figsize: 图像大小
    """
    num_nodes = causal_matrix.shape[0]
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点
    for i in range(num_nodes):
        G.add_node(i)
    
    # 添加边 (仅添加超过阈值的边)
    edge_weights = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and causal_matrix[i, j] > threshold:
                G.add_edge(j, i, weight=causal_matrix[i, j])
                edge_weights.append(causal_matrix[i, j])
    
    if len(edge_weights) == 0:
        print("⚠️ 警告: 没有边超过阈值，降低阈值重试...")
        threshold = threshold / 2
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and causal_matrix[i, j] > threshold:
                    G.add_edge(j, i, weight=causal_matrix[i, j])
                    edge_weights.append(causal_matrix[i, j])
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ========== 左图: 因果关系网络图 ==========
    ax1 = axes[0]
    
    # 使用 spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 计算节点大小 (基于入度 - 被影响程度)
    in_degrees = dict(G.in_degree())
    node_sizes = [300 + in_degrees.get(n, 0) * 100 for n in G.nodes()]
    
    # 计算节点颜色 (基于出度 - 影响力)
    out_degrees = dict(G.out_degree())
    max_out = max(out_degrees.values()) if out_degrees else 1
    node_colors = [out_degrees.get(n, 0) / max_out for n in G.nodes()]
    
    # 绘制节点
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax1, 
                                   node_size=node_sizes,
                                   node_color=node_colors,
                                   cmap=plt.cm.YlOrRd,
                                   alpha=0.8)
    
    # 节点标签
    if node_labels is None:
        node_labels = {i: f"Bus {i}" for i in range(num_nodes)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax1, font_size=8)
    
    # 绘制边 (颜色和宽度基于权重)
    if len(edge_weights) > 0:
        edges = G.edges(data=True)
        weights = [d['weight'] for _, _, d in edges]
        max_weight = max(weights)
        
        edge_colors = [w / max_weight for w in weights]
        edge_widths = [0.5 + 3 * (w / max_weight) for w in weights]
        
        nx.draw_networkx_edges(G, pos, ax=ax1,
                               edge_color=edge_colors,
                               edge_cmap=plt.cm.Blues,
                               width=edge_widths,
                               alpha=0.7,
                               arrows=True,
                               arrowsize=15,
                               connectionstyle="arc3,rad=0.1")
    
    ax1.set_title("神经格兰杰因果图\n(边: 因果方向, 颜色深度: 因果强度)", fontsize=12)
    ax1.axis('off')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                                norm=plt.Normalize(vmin=0, vmax=max_out))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, shrink=0.5, label='因果影响力 (出度)')
    
    # ========== 右图: 因果矩阵热力图 ==========
    ax2 = axes[1]
    
    im = ax2.imshow(causal_matrix, cmap='RdYlBu_r', aspect='auto')
    ax2.set_title("格兰杰因果矩阵\n(值越大表示因果影响越强)", fontsize=12)
    ax2.set_xlabel("原因节点 (Cause Node)")
    ax2.set_ylabel("结果节点 (Effect Node)")
    
    # 添加颜色条
    cbar2 = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar2.set_label('因果强度')
    
    # 添加节点刻度
    if num_nodes <= 30:
        ax2.set_xticks(range(num_nodes))
        ax2.set_yticks(range(num_nodes))
        ax2.set_xticklabels(range(num_nodes), fontsize=6)
        ax2.set_yticklabels(range(num_nodes), fontsize=6)
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 因果图已保存到: {save_path}")
    
    plt.show()
    
    return G


def analyze_causal_structure(causal_matrix, top_k=10):
    """
    分析因果结构，找出最重要的因果关系
    
    Args:
        causal_matrix: 因果权重矩阵
        top_k: 返回前 k 个最强因果关系
        
    Returns:
        causal_pairs: 按强度排序的因果关系列表 [(cause, effect, strength), ...]
    """
    num_nodes = causal_matrix.shape[0]
    
    # 找出所有非对角线元素
    causal_pairs = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                causal_pairs.append((j, i, causal_matrix[i, j]))  # (cause, effect, strength)
    
    # 按强度排序
    causal_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\n" + "=" * 50)
    print("📊 格兰杰因果分析结果")
    print("=" * 50)
    
    print(f"\n🔝 Top {top_k} 最强因果关系:")
    for idx, (cause, effect, strength) in enumerate(causal_pairs[:top_k]):
        print(f"   {idx+1}. Bus {cause} → Bus {effect} (强度: {strength:.4f})")
    
    # 统计每个节点的因果影响力 (出度)
    out_influence = np.sum(causal_matrix, axis=0)
    top_causes = np.argsort(out_influence)[::-1][:5]
    
    print(f"\n🎯 最具影响力的节点 (因果源):")
    for idx, node in enumerate(top_causes):
        print(f"   {idx+1}. Bus {node} (总影响力: {out_influence[node]:.4f})")
    
    # 统计每个节点被影响的程度 (入度)
    in_influence = np.sum(causal_matrix, axis=1)
    top_effects = np.argsort(in_influence)[::-1][:5]
    
    print(f"\n🎯 最易受影响的节点 (因果汇):")
    for idx, node in enumerate(top_effects):
        print(f"   {idx+1}. Bus {node} (被影响总量: {in_influence[node]:.4f})")
    
    # 计算稀疏度
    sparsity = np.sum(causal_matrix > 0.1) / (num_nodes * num_nodes - num_nodes)
    print(f"\n📈 因果矩阵稀疏度: {sparsity*100:.2f}% (阈值=0.1)")
    
    # 保存统计信息到文件
    stats_path = "img/causal_network_stats.txt"
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("格兰杰因果分析结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Top {top_k} 最强因果关系:\n")
        for idx, (cause, effect, strength) in enumerate(causal_pairs[:top_k]):
            f.write(f"  {idx+1}. Bus {cause} → Bus {effect} (强度: {strength:.4f})\n")
        
        f.write(f"\n最具影响力的节点 (因果源):\n")
        for idx, node in enumerate(top_causes):
            f.write(f"  {idx+1}. Bus {node} (总影响力: {out_influence[node]:.4f})\n")
        
        f.write(f"\n最易受影响的节点 (因果汇):\n")
        for idx, node in enumerate(top_effects):
            f.write(f"  {idx+1}. Bus {node} (被影响总量: {in_influence[node]:.4f})\n")
        
        f.write(f"\n因果矩阵稀疏度: {sparsity*100:.2f}% (阈值=0.1)\n")
    
    print(f"📄 统计信息已保存到: {stats_path}")
    
    return causal_pairs
