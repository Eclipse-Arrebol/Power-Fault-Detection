"""evaluation/evaluate_causal_metrics.py

因果指标评估脚本：
- 读取因果矩阵 (img/*.npy)
- 根据阈值构建因果有向图
- 计算可用于论文的结构性指标
- 计算与物理拓扑(edges.csv)的一致性(Topological Consistency)

使用:
  python .\evaluation\evaluate_causal_metrics.py
  python .\evaluation\evaluate_causal_metrics.py --matrix img/causal_matrix.npy --threshold 0.1
  python .\evaluation\evaluate_causal_metrics.py --matrix img/causal_gcn_lstm_matrix.npy --topk 50

输出:
  - img/causal_metrics_report.txt (默认)

注：本脚本只评估“因果结构/网络”本身，不涉及分类指标。
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx


@dataclass
class CausalMetricsResult:
    name: str
    num_nodes: int
    threshold: float
    topk: int
    num_edges: int
    edge_density: float
    reciprocity: float
    avg_in_degree: float
    avg_out_degree: float
    max_in_degree: int
    max_out_degree: int
    strength_out_mean: float
    strength_in_mean: float
    strength_out_p90: float
    strength_in_p90: float
    weight_mean: float
    weight_p90: float
    weight_p99: float
    gini: float
    entropy: float
    scc_count: int
    scc_max_size: int
    topo_precision: float
    topo_recall: float
    topo_f1: float
    topk_mean: float
    topk_p90: float
    top_sources: List[int]
    top_sinks: List[int]
    top_pagerank: List[int]


def _safe_float(x: float) -> float:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return 0.0
    return float(x)


def _flatten_upper_non_diag(w: np.ndarray) -> np.ndarray:
    n = w.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return w[mask].astype(float)


def gini_coefficient(values: np.ndarray) -> float:
    """Gini 系数，用于描述权重分布不均衡程度。

    参考定义：对非负数组更直观。这里会对负值取绝对值。
    """
    v = np.asarray(values, dtype=float)
    v = np.abs(v)
    v = v[v > 0]
    if v.size == 0:
        return 0.0
    v_sorted = np.sort(v)
    n = v_sorted.size
    cum = np.cumsum(v_sorted)
    g = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return _safe_float(g)


def entropy(values: np.ndarray, eps: float = 1e-12) -> float:
    """权重分布熵（越大表示越均匀）。"""
    v = np.asarray(values, dtype=float)
    v = np.abs(v)
    v = v[v > 0]
    if v.size == 0:
        return 0.0
    p = v / (np.sum(v) + eps)
    h = -np.sum(p * np.log(p + eps))
    # 归一化到 [0,1] 便于表格比较
    h_norm = h / (math.log(len(p) + eps))
    return _safe_float(h_norm)


def build_causal_graph(causal_matrix: np.ndarray, threshold: float) -> nx.DiGraph:
    n = causal_matrix.shape[0]
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = float(causal_matrix[i, j])
            if w > threshold:
                # 方向: j -> i (j cause, i effect)
                g.add_edge(j, i, weight=w)
    return g


def load_physical_edges(edges_csv_path: str) -> Tuple[set[Tuple[int, int]], set[frozenset[int]]]:
    """读取物理拓扑边。

    返回:
      - directed_edges: {(u,v), (v,u)} 双向展开
      - undirected_edges: {frozenset({u,v})}
    """
    df = pd.read_csv(edges_csv_path)
    if 'from_bus' not in df.columns or 'to_bus' not in df.columns:
        raise ValueError("edges.csv 缺少 from_bus/to_bus 列")

    directed: set[Tuple[int, int]] = set()
    undirected: set[frozenset[int]] = set()
    for u, v in zip(df['from_bus'].astype(int).values, df['to_bus'].astype(int).values):
        directed.add((u, v))
        directed.add((v, u))
        undirected.add(frozenset((u, v)))
    return directed, undirected


def topo_consistency(
    causal_g: nx.DiGraph,
    physical_directed: set[Tuple[int, int]],
    physical_undirected: set[frozenset[int]],
) -> Tuple[float, float, float]:
    """因果边与物理拓扑一致性。

    定义：因果边 (u->v) 如果 {u,v} 在物理边集合中，就认为一致。

    注意：因果边是“有向边”计数；为避免 Recall > 1，这里用“有向物理边集合”的大小作为分母。

    - precision = 一致的因果边 / 因果边总数
    - recall = 一致的因果边 / 物理有向边总数 (≈ 2 * 线路数)
    """
    causal_edges = list(causal_g.edges())
    if len(causal_edges) == 0:
        return 0.0, 0.0, 0.0

    consistent = 0
    for u, v in causal_edges:
        if (u, v) in physical_directed or frozenset((u, v)) in physical_undirected:
            consistent += 1

    precision = consistent / max(len(causal_edges), 1)
    recall = consistent / max(len(physical_directed), 1)
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return _safe_float(precision), _safe_float(recall), _safe_float(f1)


def topk_edge_stats(causal_matrix: np.ndarray, topk: int) -> Tuple[float, float]:
    weights = _flatten_upper_non_diag(causal_matrix)
    if weights.size == 0:
        return 0.0, 0.0
    weights = np.asarray(weights, dtype=float)
    # 只看正权重
    weights = weights[weights > 0]
    if weights.size == 0:
        return 0.0, 0.0
    k = min(int(topk), int(weights.size))
    top = np.sort(weights)[-k:]
    return _safe_float(float(np.mean(top))), _safe_float(float(np.percentile(top, 90)))


def compute_metrics(
    causal_matrix: np.ndarray,
    threshold: float,
    physical_directed: set[Tuple[int, int]],
    physical_undirected: set[frozenset[int]],
    name: str,
    topk: int,
) -> CausalMetricsResult:
    n = int(causal_matrix.shape[0])
    g = build_causal_graph(causal_matrix, threshold=threshold)

    num_edges = g.number_of_edges()
    edge_density = num_edges / max(n * (n - 1), 1)

    reciprocity = _safe_float(nx.reciprocity(g) or 0.0)

    in_degrees = np.array([d for _, d in g.in_degree()], dtype=int)
    out_degrees = np.array([d for _, d in g.out_degree()], dtype=int)

    # 强度（加权度）
    out_strength = {}
    in_strength = {}
    for node in g.nodes():
        out_strength[node] = float(sum(g[node][nbr]['weight'] for nbr in g.successors(node)))
        in_strength[node] = float(sum(g[pred][node]['weight'] for pred in g.predecessors(node)))

    out_strength_arr = np.array(list(out_strength.values()), dtype=float)
    in_strength_arr = np.array(list(in_strength.values()), dtype=float)

    edge_weights = np.array([d.get('weight', 0.0) for _, _, d in g.edges(data=True)], dtype=float)
    if edge_weights.size == 0:
        edge_weights = np.array([0.0], dtype=float)

    weight_mean = float(np.mean(edge_weights))
    weight_p90 = float(np.percentile(edge_weights, 90))
    weight_p99 = float(np.percentile(edge_weights, 99))

    gini = gini_coefficient(edge_weights)
    h = entropy(edge_weights)

    # SCC
    sccs = list(nx.strongly_connected_components(g))
    scc_count = len(sccs)
    scc_max_size = max((len(s) for s in sccs), default=0)

    topo_p, topo_r, topo_f1 = topo_consistency(g, physical_directed, physical_undirected)

    topk_mean, topk_p90 = topk_edge_stats(causal_matrix, topk=topk)

    # PageRank + 关键节点
    try:
        pr = nx.pagerank(g, weight='weight')
    except Exception:
        pr = {i: 1.0 / max(n, 1) for i in range(n)}

    top_sources = [k for k, _ in sorted(out_strength.items(), key=lambda x: x[1], reverse=True)[:5]]
    top_sinks = [k for k, _ in sorted(in_strength.items(), key=lambda x: x[1], reverse=True)[:5]]
    top_pagerank = [k for k, _ in sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]]

    return CausalMetricsResult(
        name=name,
        num_nodes=n,
        threshold=float(threshold),
        topk=int(topk),
        num_edges=int(num_edges),
        edge_density=_safe_float(edge_density),
        reciprocity=_safe_float(reciprocity),
        avg_in_degree=_safe_float(float(np.mean(in_degrees)) if in_degrees.size else 0.0),
        avg_out_degree=_safe_float(float(np.mean(out_degrees)) if out_degrees.size else 0.0),
        max_in_degree=int(np.max(in_degrees)) if in_degrees.size else 0,
        max_out_degree=int(np.max(out_degrees)) if out_degrees.size else 0,
        strength_out_mean=_safe_float(float(np.mean(out_strength_arr)) if out_strength_arr.size else 0.0),
        strength_in_mean=_safe_float(float(np.mean(in_strength_arr)) if in_strength_arr.size else 0.0),
        strength_out_p90=_safe_float(float(np.percentile(out_strength_arr, 90)) if out_strength_arr.size else 0.0),
        strength_in_p90=_safe_float(float(np.percentile(in_strength_arr, 90)) if in_strength_arr.size else 0.0),
        weight_mean=_safe_float(weight_mean),
        weight_p90=_safe_float(weight_p90),
        weight_p99=_safe_float(weight_p99),
        gini=_safe_float(gini),
        entropy=_safe_float(h),
        scc_count=int(scc_count),
        scc_max_size=int(scc_max_size),
        topo_precision=_safe_float(topo_p),
        topo_recall=_safe_float(topo_r),
        topo_f1=_safe_float(topo_f1),
        topk_mean=_safe_float(topk_mean),
        topk_p90=_safe_float(topk_p90),
        top_sources=top_sources,
        top_sinks=top_sinks,
        top_pagerank=top_pagerank,
    )


def format_report(results: Sequence[CausalMetricsResult]) -> str:
    lines: List[str] = []
    lines.append("=" * 90)
    lines.append("因果结构指标评估报告 (Causal Metrics Report)")
    lines.append("=" * 90)

    for r in results:
        lines.append("")
        lines.append("-" * 90)
        lines.append(f"模型/矩阵: {r.name}")
        lines.append(f"节点数 N: {r.num_nodes}")
        lines.append(f"阈值 τ: {r.threshold}")
        lines.append(f"边数 |E|: {r.num_edges}")
        lines.append(f"边密度 d=|E|/(N(N-1)): {r.edge_density:.6f}")
        lines.append(f"互惠性 Reciprocity: {r.reciprocity:.4f}")
        lines.append(f"平均入度/出度: {r.avg_in_degree:.2f} / {r.avg_out_degree:.2f}")
        lines.append(f"最大入度/出度: {r.max_in_degree} / {r.max_out_degree}")
        lines.append(f"出强度(mean/p90): {r.strength_out_mean:.4f} / {r.strength_out_p90:.4f}")
        lines.append(f"入强度(mean/p90): {r.strength_in_mean:.4f} / {r.strength_in_p90:.4f}")
        lines.append(f"边权重(mean/p90/p99): {r.weight_mean:.4f} / {r.weight_p90:.4f} / {r.weight_p99:.4f}")
        lines.append(f"权重不均衡(Gini): {r.gini:.4f}")
        lines.append(f"权重熵(归一化Entropy): {r.entropy:.4f}")
        lines.append(f"SCC数量/最大SCC大小: {r.scc_count} / {r.scc_max_size}")
        lines.append("物理拓扑一致性(用 edges.csv):")
        lines.append(f"  Precision: {r.topo_precision:.4f}")
        lines.append(f"  Recall:    {r.topo_recall:.4f}")
        lines.append(f"  F1:        {r.topo_f1:.4f}")
        lines.append(f"TopK({r.topk})因果强度(mean/p90): {r.topk_mean:.4f} / {r.topk_p90:.4f}")
        lines.append(f"Top-5 因果源(出强度最大): {r.top_sources}")
        lines.append(f"Top-5 因果汇(入强度最大): {r.top_sinks}")
        lines.append(f"Top-5 PageRank: {r.top_pagerank}")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=str,
        default=None,
        help="因果矩阵路径 .npy (默认评估 img/ 下常见文件)",
    )
    parser.add_argument("--threshold", type=float, default=0.1, help="阈值 τ")
    parser.add_argument("--topk", type=int, default=50, help="Top-K 因果边强度统计")
    parser.add_argument(
        "--edges",
        type=str,
        default="dataset/edges.csv",
        help="物理拓扑边文件路径",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="img/causal_metrics_report.txt",
        help="输出报告路径",
    )

    args = parser.parse_args()

    if not os.path.exists(args.edges):
        raise FileNotFoundError(f"找不到物理拓扑文件: {args.edges}")

    physical_directed, physical_undirected = load_physical_edges(args.edges)

    matrices: List[Tuple[str, str]] = []
    if args.matrix is not None:
        matrices.append((os.path.basename(args.matrix), args.matrix))
    else:
        # 默认扫描 img/ 下常见矩阵
        candidates = [
            "img/causal_matrix.npy",
            "img/causal_gcn_lstm_matrix.npy",
        ]
        for p in candidates:
            if os.path.exists(p):
                matrices.append((os.path.basename(p), p))

    if not matrices:
        raise FileNotFoundError("未找到因果矩阵 .npy，请用 --matrix 指定")

    results: List[CausalMetricsResult] = []
    for name, path in matrices:
        w = np.load(path)
        if w.ndim != 2 or w.shape[0] != w.shape[1]:
            raise ValueError(f"因果矩阵形状非法: {path}, shape={w.shape}")
        results.append(
            compute_metrics(
                causal_matrix=w,
                threshold=float(args.threshold),
                physical_directed=physical_directed,
                physical_undirected=physical_undirected,
                name=name,
                topk=int(args.topk),
            )
        )

    report = format_report(results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ 因果指标报告已生成: {args.out}")


if __name__ == "__main__":
    main()
