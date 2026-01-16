"""
神经格兰杰因果网络可视化工具

功能：
1. 读取训练好的因果矩阵
2. 构建有向网络拓扑图
3. 可视化因果传播路径

使用方法：
    python visualize_causal_network.py

输出：
    - img/causal_network_topology.png: 网络拓扑图
    - img/causal_network_circular.png: 环形布局图
    - img/causal_network_spring.png: 弹簧布局图
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CausalNetworkVisualizer:
    """因果网络可视化器"""
    
    def __init__(self, causal_matrix, threshold=0.1):
        """
        初始化
        
        Args:
            causal_matrix: 因果矩阵 [N, N]，matrix[i,j] 表示节点j对节点i的影响
            threshold: 因果强度阈值，低于此值的边不显示
        """
        self.causal_matrix = causal_matrix
        self.threshold = threshold
        self.num_nodes = causal_matrix.shape[0]
        
        # 创建有向图
        self.G = nx.DiGraph()
        self._build_graph()
        
    def _build_graph(self):
        """构建有向图"""
        print(f">>> 构建因果网络图 (节点数: {self.num_nodes}, 阈值: {self.threshold})...")
        
        # 添加所有节点
        self.G.add_nodes_from(range(self.num_nodes))
        
        # 添加边（仅添加超过阈值的因果关系）
        edge_count = 0
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and self.causal_matrix[i, j] > self.threshold:
                    # 边的方向: j -> i (节点j对节点i有因果影响)
                    weight = float(self.causal_matrix[i, j])
                    self.G.add_edge(j, i, weight=weight)
                    edge_count += 1
        
        print(f"    - 有效边数: {edge_count}")
        print(f"    - 平均度数: {edge_count / self.num_nodes:.2f}")
        
        # 计算节点统计信息
        self._compute_node_stats()
        
    def _compute_node_stats(self):
        """计算节点统计信息"""
        # 出度（影响力）：该节点影响了多少其他节点
        self.out_degrees = dict(self.G.out_degree())
        
        # 入度（被影响程度）：该节点被多少其他节点影响
        self.in_degrees = dict(self.G.in_degree())
        
        # 出强度（总影响力）：对其他节点的因果影响总和
        self.out_strength = {}
        for node in self.G.nodes():
            strength = sum([self.G[node][neighbor]['weight'] 
                           for neighbor in self.G.successors(node)])
            self.out_strength[node] = strength
        
        # 入强度（总被影响程度）：被其他节点的因果影响总和
        self.in_strength = {}
        for node in self.G.nodes():
            strength = sum([self.G[predecessor][node]['weight'] 
                           for predecessor in self.G.predecessors(node)])
            self.in_strength[node] = strength
        
        # PageRank（节点重要性）
        try:
            self.pagerank = nx.pagerank(self.G, weight='weight')
        except:
            self.pagerank = {n: 1.0 for n in self.G.nodes()}
        
        # 找出关键节点
        self._find_key_nodes()
        
    def _find_key_nodes(self):
        """找出关键节点"""
        # Top-5 影响力节点（出强度最大）
        sorted_out = sorted(self.out_strength.items(), key=lambda x: x[1], reverse=True)
        self.top_sources = [node for node, _ in sorted_out[:5]]
        
        # Top-5 易受影响节点（入强度最大）
        sorted_in = sorted(self.in_strength.items(), key=lambda x: x[1], reverse=True)
        self.top_sinks = [node for node, _ in sorted_in[:5]]
        
        # Top-5 PageRank 节点
        sorted_pr = sorted(self.pagerank.items(), key=lambda x: x[1], reverse=True)
        self.top_pagerank = [node for node, _ in sorted_pr[:5]]
        
        print(f"\n>>> 关键节点分析:")
        print(f"    - Top-5 因果源 (最具影响力): {self.top_sources}")
        print(f"    - Top-5 因果汇 (最易受影响): {self.top_sinks}")
        print(f"    - Top-5 PageRank (最重要): {self.top_pagerank}")
        
    def plot_network(self, layout='spring', figsize=(16, 12), 
                     save_path='img/causal_network_topology.png'):
        """
        绘制网络拓扑图
        
        Args:
            layout: 布局算法 ('spring', 'circular', 'kamada_kawai', 'shell')
            figsize: 图像大小
            save_path: 保存路径
        """
        print(f"\n>>> 绘制网络拓扑图 (布局: {layout})...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 选择布局算法
        if layout == 'spring':
            pos = nx.spring_layout(self.G, k=2, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(self.G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.G)
        elif layout == 'shell':
            pos = nx.shell_layout(self.G)
        else:
            pos = nx.spring_layout(self.G, seed=42)
        
        # ========== 节点样式 ==========
        # 节点大小：基于出强度（影响力）
        max_out_strength = max(self.out_strength.values()) if self.out_strength else 1.0
        node_sizes = [300 + 2000 * (self.out_strength.get(n, 0) / max_out_strength) 
                      for n in self.G.nodes()]
        
        # 节点颜色：基于 PageRank（重要性）
        max_pr = max(self.pagerank.values()) if self.pagerank else 1.0
        node_colors = [self.pagerank.get(n, 0) / max_pr for n in self.G.nodes()]
        
        # 绘制节点
        nodes = nx.draw_networkx_nodes(
            self.G, pos, ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.YlOrRd,
            alpha=0.8,
            edgecolors='black',
            linewidths=1.5
        )
        
        # 标记关键节点
        highlight_nodes = set(self.top_sources + self.top_pagerank)
        highlight_sizes = [node_sizes[i] * 1.2 for i, n in enumerate(self.G.nodes()) 
                          if n in highlight_nodes]
        nx.draw_networkx_nodes(
            self.G, pos, ax=ax,
            nodelist=list(highlight_nodes),
            node_size=highlight_sizes,
            node_color='red',
            alpha=0.3,
            edgecolors='darkred',
            linewidths=3
        )
        
        # ========== 边样式 ==========
        # 获取所有边的权重
        edges = self.G.edges(data=True)
        weights = [d['weight'] for _, _, d in edges]
        max_weight = max(weights) if weights else 1.0
        
        # 边的宽度和颜色
        edge_widths = [0.5 + 3 * (w / max_weight) for w in weights]
        edge_colors = [w / max_weight for w in weights]
        
        # 绘制边
        nx.draw_networkx_edges(
            self.G, pos, ax=ax,
            edge_color=edge_colors,
            edge_cmap=plt.cm.Blues,
            width=edge_widths,
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            node_size=node_sizes
        )
        
        # ========== 节点标签 ==========
        # 只标注关键节点
        labels = {n: f"Bus{n}" for n in highlight_nodes}
        nx.draw_networkx_labels(
            self.G, pos, labels=labels, ax=ax,
            font_size=10,
            font_color='darkred',
            font_weight='bold'
        )
        
        # ========== 图例和标题 ==========
        ax.set_title(
            f"神经格兰杰因果网络拓扑图\n"
            f"节点数: {self.num_nodes} | 边数: {len(self.G.edges())} | 阈值: {self.threshold}",
            fontsize=16, fontweight='bold', pad=20
        )
        
        # 添加颜色条
        sm_nodes = plt.cm.ScalarMappable(
            cmap=plt.cm.YlOrRd, 
            norm=plt.Normalize(vmin=0, vmax=max_pr)
        )
        sm_nodes.set_array([])
        cbar_nodes = plt.colorbar(sm_nodes, ax=ax, shrink=0.5, pad=0.02)
        cbar_nodes.set_label('节点重要性 (PageRank)', fontsize=10)
        
        # 图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='red', markersize=15, alpha=0.8,
                      label='关键节点 (高影响力)'),
            plt.Line2D([0], [0], color='blue', linewidth=3, alpha=0.6,
                      label='强因果关系'),
            plt.Line2D([0], [0], color='blue', linewidth=1, alpha=0.6,
                      label='弱因果关系')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"    - 图像已保存: {save_path}")
        plt.show()
        
    def plot_hierarchical_network(self, save_path='img/causal_network_hierarchical.png'):
        """
        绘制层次化网络图（按影响力分层）
        """
        print(f"\n>>> 绘制层次化网络图...")
        
        fig, ax = plt.subplots(figsize=(18, 12))
        
        # 按出强度分层
        sorted_nodes = sorted(self.out_strength.items(), key=lambda x: x[1], reverse=True)
        
        # 分成3层：高影响力、中等影响力、低影响力
        n_high = len(sorted_nodes) // 3
        n_mid = len(sorted_nodes) // 3
        
        high_influence = [n for n, _ in sorted_nodes[:n_high]]
        mid_influence = [n for n, _ in sorted_nodes[n_high:n_high+n_mid]]
        low_influence = [n for n, _ in sorted_nodes[n_high+n_mid:]]
        
        # 手动设置层次化布局
        pos = {}
        
        # 第1层（顶部）：高影响力节点
        for i, node in enumerate(high_influence):
            x = (i - len(high_influence)/2) * 0.8
            y = 2.0
            pos[node] = (x, y)
        
        # 第2层（中部）：中等影响力节点
        for i, node in enumerate(mid_influence):
            x = (i - len(mid_influence)/2) * 0.6
            y = 0.0
            pos[node] = (x, y)
        
        # 第3层（底部）：低影响力节点
        for i, node in enumerate(low_influence):
            x = (i - len(low_influence)/2) * 0.5
            y = -2.0
            pos[node] = (x, y)
        
        # 节点样式
        max_out_strength = max(self.out_strength.values())
        node_sizes = [300 + 1500 * (self.out_strength.get(n, 0) / max_out_strength) 
                      for n in self.G.nodes()]
        
        # 分层绘制节点
        nx.draw_networkx_nodes(
            self.G, pos, ax=ax,
            nodelist=high_influence,
            node_size=[node_sizes[n] for n in high_influence],
            node_color='red', alpha=0.8, label='高影响力层'
        )
        
        nx.draw_networkx_nodes(
            self.G, pos, ax=ax,
            nodelist=mid_influence,
            node_size=[node_sizes[n] for n in mid_influence],
            node_color='orange', alpha=0.8, label='中等影响力层'
        )
        
        nx.draw_networkx_nodes(
            self.G, pos, ax=ax,
            nodelist=low_influence,
            node_size=[node_sizes[n] for n in low_influence],
            node_color='lightblue', alpha=0.8, label='低影响力层'
        )
        
        # 绘制边
        edges = self.G.edges(data=True)
        weights = [d['weight'] for _, _, d in edges]
        max_weight = max(weights) if weights else 1.0
        edge_widths = [0.5 + 2 * (w / max_weight) for w in weights]
        
        nx.draw_networkx_edges(
            self.G, pos, ax=ax,
            width=edge_widths,
            alpha=0.4,
            arrows=True,
            arrowsize=12,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1'
        )
        
        # 标注 Top-5 节点
        labels = {n: f"Bus{n}" for n in self.top_sources[:5]}
        nx.draw_networkx_labels(self.G, pos, labels=labels, ax=ax, font_size=9)
        
        ax.set_title(
            "层次化因果网络图\n(节点按影响力分层排列)",
            fontsize=16, fontweight='bold', pad=20
        )
        ax.legend(loc='upper right', fontsize=12)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"    - 图像已保存: {save_path}")
        plt.show()
        
    def export_graph_stats(self, save_path='img/causal_network_stats.txt'):
        """导出网络统计信息"""
        print(f"\n>>> 导出网络统计信息...")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("神经格兰杰因果网络统计报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本信息
            f.write(f"节点数: {self.num_nodes}\n")
            f.write(f"边数: {len(self.G.edges())}\n")
            f.write(f"因果阈值: {self.threshold}\n")
            f.write(f"平均度数: {len(self.G.edges()) / self.num_nodes:.2f}\n\n")
            
            # Top-10 因果源
            f.write("-" * 60 + "\n")
            f.write("Top-10 因果源节点 (最具影响力)\n")
            f.write("-" * 60 + "\n")
            sorted_out = sorted(self.out_strength.items(), key=lambda x: x[1], reverse=True)
            for rank, (node, strength) in enumerate(sorted_out[:10], 1):
                f.write(f"{rank:2d}. Bus {node:3d} - 影响强度: {strength:.4f}, "
                       f"出度: {self.out_degrees[node]}\n")
            
            # Top-10 因果汇
            f.write("\n" + "-" * 60 + "\n")
            f.write("Top-10 因果汇节点 (最易受影响)\n")
            f.write("-" * 60 + "\n")
            sorted_in = sorted(self.in_strength.items(), key=lambda x: x[1], reverse=True)
            for rank, (node, strength) in enumerate(sorted_in[:10], 1):
                f.write(f"{rank:2d}. Bus {node:3d} - 受影响强度: {strength:.4f}, "
                       f"入度: {self.in_degrees[node]}\n")
            
            # Top-10 PageRank
            f.write("\n" + "-" * 60 + "\n")
            f.write("Top-10 PageRank 节点 (最重要)\n")
            f.write("-" * 60 + "\n")
            sorted_pr = sorted(self.pagerank.items(), key=lambda x: x[1], reverse=True)
            for rank, (node, pr) in enumerate(sorted_pr[:10], 1):
                f.write(f"{rank:2d}. Bus {node:3d} - PageRank: {pr:.6f}\n")
            
            # 网络密度
            density = nx.density(self.G)
            f.write("\n" + "-" * 60 + "\n")
            f.write(f"网络密度: {density:.4f}\n")
            
            # 连通性分析
            if nx.is_weakly_connected(self.G):
                f.write("弱连通性: 是（所有节点可达）\n")
            else:
                n_components = nx.number_weakly_connected_components(self.G)
                f.write(f"弱连通性: 否（{n_components} 个连通分量）\n")
            
        print(f"    - 统计信息已保存: {save_path}")


def main():
    """主函数"""
    import os
    
    # 创建输出目录
    os.makedirs("img", exist_ok=True)
    
    # 加载因果矩阵
    print("=" * 60)
    print("神经格兰杰因果网络可视化")
    print("=" * 60)
    
    causal_matrix_path = "img/causal_matrix.npy"
    
    if not os.path.exists(causal_matrix_path):
        print(f"\n❌ 错误: 找不到因果矩阵文件 {causal_matrix_path}")
        print("   请先运行 train.py 并设置 USE_NGC=True 训练模型")
        return
    
    print(f"\n>>> 加载因果矩阵: {causal_matrix_path}")
    causal_matrix = np.load(causal_matrix_path)
    print(f"    - 矩阵形状: {causal_matrix.shape}")
    print(f"    - 最大值: {causal_matrix.max():.4f}")
    print(f"    - 最小值: {causal_matrix.min():.4f}")
    print(f"    - 平均值: {causal_matrix.mean():.4f}")
    
    # 创建可视化器
    threshold = 0.5  # 可以调整这个阈值
    visualizer = CausalNetworkVisualizer(causal_matrix, threshold=threshold)
    
    # 绘制不同布局的网络图
    print("\n" + "=" * 60)
    visualizer.plot_network(layout='spring', 
                            save_path='img/causal_network_spring.png')
    
    visualizer.plot_network(layout='circular', 
                            save_path='img/causal_network_circular.png')
    
    # 绘制层次化网络图
    visualizer.plot_hierarchical_network()
    
    # 导出统计信息
    visualizer.export_graph_stats()
    
    print("\n" + "=" * 60)
    print("✅ 可视化完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - img/causal_network_spring.png      (弹簧布局)")
    print("  - img/causal_network_circular.png    (环形布局)")
    print("  - img/causal_network_hierarchical.png (层次布局)")
    print("  - img/causal_network_stats.txt       (统计报告)")
    print()


if __name__ == "__main__":
    main()
