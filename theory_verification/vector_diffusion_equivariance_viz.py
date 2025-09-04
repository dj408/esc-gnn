"""
Vector diffusion visualization and rotational equivariance demo.

This script:
- Generates a random 2D vector field on N points in [-1, 1]^2
- Builds a symmetric, undirected, optionally-weighted k-NN graph
- Reuses process_pyg_data to construct the sparse vector diffusion operator Q
- Applies Q to the vector features t steps
- Visualizes diffusion across panels and demonstrates rotational equivariance:
  top row: diffuse first (t steps), then rotate the final frame by 90 degrees
  bottom row: rotate first, then diffuse (t steps)

Run:
  python theory_verification/vector_diffusion_equivariance_viz.py \
      --viz-kind dyadic --num-nodes 50 --k 3 --t 3 --weighted --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Literal

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected
import matplotlib
matplotlib.use("Agg")  # Ensure headless rendering
import matplotlib.pyplot as plt
# Set LaTeX-like fonts (no TeX install required)
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = [
    "CMU Serif",
    "Computer Modern Roman",
    "STIX",
    "DejaVu Serif",
]

# Ensure project root is on sys.path for imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_processing.process_pyg_data import process_pyg_data
from pyg_utilities import get_Batch_Wjxs


# -------------------------------
# Configuration
# -------------------------------
@dataclass
class VizConfig:
    num_nodes: int = 50
    k: int = 3
    t: int = 3
    weighted: bool = True
    gaussian_eps: float = 0.5
    r_cutoff: float = 0.9
    seed: int = 0
    device: str = "cpu"
    central_color: str = "blue"
    neighbor_frontier_color: str = "black"
    visited_layer_color: str = "#6fa8dc"  # light blue
    default_arrow_color: str = "#cccccc"   # light gray
    rotate_background_color: str = "#f5fbff"
    quiver_scale: Optional[float] = 1.0  # Use data units for vector lengths
    vector_min_mag: float = 0.4
    vector_max_mag: float = 1.0
    central_length_min_fraction: float = 0.6  # min fraction of axis range for center arrow
    plot_title: Optional[str] = None
    viz_kind: Literal["powers", "dyadic"] = "powers"

# -------------------------------
# Utilities
# -------------------------------
def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def generate_positions_and_vectors(
    num_nodes: int,
    vector_min_mag: float,
    vector_max_mag: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random 2D positions in [-1,1]^2 and random 2D vectors.
    """
    xy = np.random.uniform(-1.0, 1.0, size=(num_nodes, 2))
    angles = np.random.uniform(0.0, 2.0 * np.pi, size=(num_nodes,))
    mags = np.random.uniform(vector_min_mag, vector_max_mag, size=(num_nodes,))
    vx = mags * np.cos(angles)
    vy = mags * np.sin(angles)
    v = np.stack([vx, vy], axis=1)
    return xy, v


def build_knn_undirected_graph(pos: torch.Tensor, k: int) -> torch.Tensor:
    """
    Build symmetric, undirected k-NN graph using torch_geometric.
    """
    edge_index = knn_graph(pos, k=k, loop=False)
    edge_index = to_undirected(edge_index, reduce="min")
    return edge_index


def compute_edge_weights(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    gaussian_eps: float,
    r_cutoff: float,
) -> torch.Tensor:
    """
    Compute smooth distance weights w(r) = exp(-(r/eps)^2) * 0.5 * (cos(pi r / r_c) + 1) for r < r_c; else 0.
    """
    src, dst = edge_index
    diffs = pos[src] - pos[dst]
    dists = torch.linalg.norm(diffs, dim=1)
    gauss = torch.exp(-((dists / gaussian_eps) ** 2))
    cosine_cut = 0.5 * (torch.cos(np.pi * dists / r_cutoff) + 1.0)
    weights = torch.where(dists < r_cutoff, gauss * cosine_cut, torch.zeros_like(gauss))
    return weights


def rotation_matrix_2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def rotate_array(arr: np.ndarray, R: np.ndarray) -> np.ndarray:
    return (R @ arr.T).T


def ravel_vectors(v: torch.Tensor) -> torch.Tensor:
    """
    Ravel (N, d) vectors into (N*d, 1) column vector, row-wise stacking.
    """
    N, d = v.shape
    return v.reshape(N * d, 1)


def unravel_vectors(v_vec: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Unravel (N*d, 1) back to (N, d).
    """
    return v_vec.reshape(num_nodes, -1)


def build_adjacency_dict(edge_index: torch.Tensor, num_nodes: int) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
    for s, t in edge_index.t().tolist():
        adj[s].add(t)
    return adj


def bfs_layers(adj: Dict[int, Set[int]], center: int, max_depth: int) -> List[Set[int]]:
    """
    Return BFS layers from center up to max_depth (inclusive for frontier computation).
    layers[0] = {center}; layers[1] = 1-hop; etc.
    """
    layers: List[Set[int]] = [set([center])]
    visited: Set[int] = set([center])
    frontier: Set[int] = set([center])
    for _ in range(max_depth):
        next_frontier: Set[int] = set()
        for u in frontier:
            next_frontier.update(adj[u])
        next_frontier.difference_update(visited)
        if len(next_frontier) == 0:
            layers.append(set())
            frontier = set()
            continue
        layers.append(next_frontier)
        visited.update(next_frontier)
        frontier = next_frontier
    return layers


def find_most_central_node(xy: np.ndarray) -> int:
    centroid = xy.mean(axis=0, keepdims=True)
    d = np.linalg.norm(xy - centroid, axis=1)
    return int(np.argmin(d))


def plot_vectors(
    ax: plt.Axes,
    xy: np.ndarray,
    v: np.ndarray,
    default_color: str,
    scale: Optional[float],
    highlight_center: Optional[int] = None,
    frontier_nodes: Optional[Sequence[int]] = None,
    visited_layer_nodes: Optional[Sequence[int]] = None,
    center_color: str = "blue",
    frontier_color: str = "black",
    visited_layer_color: str = "#6fa8dc",
    title: Optional[str] = None,
    facecolor: Optional[str] = None,
) -> None:
    if facecolor is not None:
        ax.set_facecolor(facecolor)

    # Prepare highlighted index sets
    N = xy.shape[0]
    visited_idx = np.array(list(visited_layer_nodes), dtype=int) if visited_layer_nodes is not None and len(visited_layer_nodes) > 0 else np.array([], dtype=int)
    frontier_idx = np.array(list(frontier_nodes), dtype=int) if frontier_nodes is not None and len(frontier_nodes) > 0 else np.array([], dtype=int)
    center_idx_array = np.array([int(highlight_center)], dtype=int) if highlight_center is not None else np.array([], dtype=int)

    # Exclude highlighted nodes from the base gray layer to avoid duplicates
    exclude_list: List[int] = []
    if visited_idx.size > 0:
        exclude_list.extend(visited_idx.tolist())
    if frontier_idx.size > 0:
        exclude_list.extend(frontier_idx.tolist())
    if center_idx_array.size > 0:
        exclude_list.extend(center_idx_array.tolist())
    if len(exclude_list) > 0:
        exclude = np.unique(np.array(exclude_list, dtype=int))
    else:
        exclude = np.array([], dtype=int)
    base_mask = np.ones(N, dtype=bool)
    if exclude.size > 0:
        base_mask[exclude] = False

    # Plot base (light gray) vectors
    if base_mask.any():
        ax.quiver(
            xy[base_mask, 0], xy[base_mask, 1], v[base_mask, 0], v[base_mask, 1],
            color=default_color, angles="xy", scale_units="xy", scale=scale, width=0.004, alpha=0.7, pivot="tail"
        )

    # Highlight visited layer nodes (light blue)
    if visited_idx.size > 0:
        ax.quiver(
            xy[visited_idx, 0], xy[visited_idx, 1], v[visited_idx, 0], v[visited_idx, 1],
            color=visited_layer_color, angles="xy", scale_units="xy", scale=scale, width=0.006, pivot="tail"
        )

    # Highlight frontier nodes (black)
    if frontier_idx.size > 0:
        ax.quiver(
            xy[frontier_idx, 0], xy[frontier_idx, 1], v[frontier_idx, 0], v[frontier_idx, 1],
            color=frontier_color, angles="xy", scale_units="xy", scale=scale, width=0.007, pivot="tail"
        )

    # Highlight center node (blue) on top
    if center_idx_array.size == 1:
        i = int(center_idx_array[0])
        ax.quiver(
            xy[i:i+1, 0], xy[i:i+1, 1], v[i:i+1, 0], v[i:i+1, 1],
            color=center_color, angles="xy", scale_units="xy", scale=scale, width=0.012, pivot="tail"
        )

    ax.set_aspect("equal")
    # Compute plot limits based on both tails and heads so long arrows are visible
    base_idx = np.where(base_mask)[0]
    all_indices = np.unique(np.concatenate([
        base_idx,
        visited_idx if visited_idx.size > 0 else np.array([], dtype=int),
        frontier_idx if frontier_idx.size > 0 else np.array([], dtype=int),
        center_idx_array if center_idx_array.size > 0 else np.array([], dtype=int),
    ]))
    if all_indices.size > 0:
        x_tail = xy[all_indices, 0]
        y_tail = xy[all_indices, 1]
        u = v[all_indices, 0]
        vv = v[all_indices, 1]
        x_heads = x_tail + u
        y_heads = y_tail + vv
        pad = 0.1
        x_min = float(np.minimum(x_tail, x_heads).min()) - pad
        x_max = float(np.maximum(x_tail, x_heads).max()) + pad
        y_min = float(np.minimum(y_tail, y_heads).min()) - pad
        y_max = float(np.maximum(y_tail, y_heads).max()) + pad
    else:
        pad = 0.1
        x_min, x_max = xy[:, 0].min() - pad, xy[:, 0].max() + pad
        y_min, y_max = xy[:, 1].min() - pad, xy[:, 1].max() + pad
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.grid(True, alpha=0.2)
    if title:
        ax.set_title(title)


# -------------------------------
# Core pipeline
# -------------------------------
def make_data(
    xy: np.ndarray,
    v: np.ndarray,
    k: int,
    weighted: bool,
    gaussian_eps: float,
    r_cutoff: float,
) -> Data:
    pos = torch.tensor(xy, dtype=torch.float32)
    vec = torch.tensor(v, dtype=torch.float32)

    edge_index = build_knn_undirected_graph(pos, k)
    data = Data(pos=pos, edge_index=edge_index)
    data["v"] = vec

    if weighted:
        w = compute_edge_weights(
            pos, 
            edge_index, 
            gaussian_eps=gaussian_eps, 
            r_cutoff=r_cutoff,
        )
        data.edge_weight = w

    return data


def compute_Q_and_diffusions(
    data: Data,
    v_init: torch.Tensor,
    t: int,
    device: str,
    vector_feat_key_for_Q: str = "pos",
) -> List[torch.Tensor]:
    """
    Return list of vector states [v^(0), v^(1), ..., v^(t)] as (N, 2) tensors.
    Q is constructed from the provided Data using `vector_feat_key_for_Q`.
    Diffusion is applied to v_init.
    """
    processed = process_pyg_data(
        data,
        vector_feat_key=vector_feat_key_for_Q,
        device=device,
        graph_construction=None,  # use the provided graph/weights
        return_data_object=True,
    )

    Q = processed.Q.coalesce().to(device)
    v0 = v_init.to(device)  # shape (N, 2)
    N = v0.shape[0]

    states: List[torch.Tensor] = [v0.detach().cpu()]
    v_vec = ravel_vectors(v0)
    for _ in range(t):
        v_vec = torch.sparse.mm(Q, v_vec)
        v_next = unravel_vectors(v_vec, N)
        states.append(v_next.detach().cpu())
    return states


def compute_Q_and_dyadic_wavelets(
    data: Data,
    v_init: torch.Tensor,
    device: str,
    *,
    J: int = 2,
    vector_feat_key_for_Q: str = "pos",
) -> List[torch.Tensor]:
    """
    Return list [x, (I-Q)x, (Q-Q^2)x, (Q^2-Q^4)x] as (N, 2) tensors.
    Uses dyadic wavelets computed via get_Batch_Wjxs on the vector diffusion operator Q.
    """
    processed = process_pyg_data(
        data,
        vector_feat_key=vector_feat_key_for_Q,
        device=device,
        graph_construction=None,
        return_data_object=True,
    )
    Q = processed.Q.coalesce().to(device)

    v0 = v_init.to(device)  # (N, 2)
    N = v0.shape[0]

    # Flatten vectors to (N*2, 1) to match operator shape
    v_vec = ravel_vectors(v0)  # (N*2, 1)
    W = get_Batch_Wjxs(
        x=v_vec,
        P_sparse=Q,
        scales_type="dyadic",
        J=J,
        include_lowpass=False,
        filter_stack_dim=-1,
    )  # shape: (N*2, 1, 3)

    # Unpack into (N, 2) arrays
    outs: List[torch.Tensor] = [v0.detach().cpu()]
    for idx in range(W.shape[-1]):
        w_vec = W[:, 0, idx].detach().cpu()
        outs.append(unravel_vectors(w_vec, N))
    return outs


def build_layer_sets(edge_index: torch.Tensor, num_nodes: int, center: int, depth: int) -> Tuple[List[Set[int]], List[Set[int]]]:
    """
    Build (visited_layers, frontier_layers) for panels.
    - visited_layers[s] := nodes at exact distance s from center
    - frontier_layers[s] := nodes at distance s+1 from center (excluding already visited)
    """
    adj = build_adjacency_dict(edge_index, num_nodes)
    layers = bfs_layers(adj, center=center, max_depth=depth + 1)  # build one extra for frontier depth
    visited_layers: List[Set[int]] = []
    frontier_layers: List[Set[int]] = []
    for s in range(depth + 1):
        visited_layers.append(layers[s] if s < len(layers) else set())
        frontier_layers.append(layers[s + 1] if (s + 1) < len(layers) else set())
    return visited_layers, frontier_layers


def next_step_nodes_from_layers(
    adj: Dict[int, Set[int]],
    visited_layers: List[Set[int]],
    s: int,
) -> Set[int]:
    """
    Compute nodes involved in the next diffusion step from layers up to s.
    This includes neighbors of all nodes in layers[0..s] and does NOT exclude
    already visited nodes.
    """
    upto_s: Set[int] = set()
    for i in range(min(s + 1, len(visited_layers))):
        upto_s.update(visited_layers[i])
    next_nodes: Set[int] = set()
    for u in upto_s:
        next_nodes.update(adj.get(u, set()))
    return next_nodes


def visualize(
    cfg: VizConfig,
) -> None:
    set_seeds(cfg.seed)

    # Generate base (unrotated) data
    xy_np, v_np = generate_positions_and_vectors(
        cfg.num_nodes, cfg.vector_min_mag, cfg.vector_max_mag
    )
    center_idx = find_most_central_node(xy_np)

    # Robustly ensure central (blue) vector is clearly visible with a long magnitude
    x_range = float(xy_np[:, 0].max() - xy_np[:, 0].min())
    y_range = float(xy_np[:, 1].max() - xy_np[:, 1].min())
    axis_range = max(x_range, y_range)
    mags = np.linalg.norm(v_np, axis=1)
    p95 = float(np.percentile(mags, 95))
    # Target magnitude: at least a fraction of the plot extent, and no more than ~view
    target_mag = max(2.0 * p95, cfg.central_length_min_fraction * axis_range)
    target_mag = min(target_mag, 0.9 * axis_range)
    center_vec = v_np[center_idx]
    norm = float(np.linalg.norm(center_vec))
    if norm > 0.0:
        v_np[center_idx] = (center_vec / norm) * target_mag
    else:
        v_np[center_idx] = np.array([target_mag, 0.0], dtype=v_np.dtype)

    # Build graph/Data and compute diffusion states or wavelets
    data_unrot = make_data(
        xy=xy_np,
        v=v_np,
        k=cfg.k,
        weighted=cfg.weighted,
        gaussian_eps=cfg.gaussian_eps,
        r_cutoff=cfg.r_cutoff,
    )
    v_init_t = torch.tensor(v_np, dtype=torch.float32)
    if cfg.viz_kind == "powers":
        states_unrot = compute_Q_and_diffusions(
            data_unrot, v_init=v_init_t, t=cfg.t, device=cfg.device, vector_feat_key_for_Q="pos"
        )
    else:
        # dyadic wavelets
        states_unrot = compute_Q_and_dyadic_wavelets(
            data_unrot, v_init=v_init_t, device=cfg.device, J=2, vector_feat_key_for_Q="pos"
        )  # [x, W1, W2, W3]

    # Layer sets for highlighting (based on the unrotated adjacency)
    if cfg.viz_kind == "powers":
        visited_layers, frontier_layers = build_layer_sets(
            data_unrot.edge_index, cfg.num_nodes, center_idx, depth=cfg.t
        )
    else:
        # For dyadic panels, we will reference s in {0,1,2,4}
        visited_layers, frontier_layers = build_layer_sets(
            data_unrot.edge_index, cfg.num_nodes, center_idx, depth=4
        )
    adj_unrot = build_adjacency_dict(data_unrot.edge_index, cfg.num_nodes)

    # Prepare rotated-first data (for bottom row)
    R90 = rotation_matrix_2d(np.pi / 2.0)
    xy_rot_np = rotate_array(xy_np, R90)
    v_rot_np = rotate_array(v_np, R90)

    data_rot_first = make_data(
        xy=xy_rot_np,
        v=v_rot_np,
        k=cfg.k,
        weighted=cfg.weighted,
        gaussian_eps=cfg.gaussian_eps,
        r_cutoff=cfg.r_cutoff,
    )
    v_rot_init_t = torch.tensor(v_rot_np, dtype=torch.float32)
    if cfg.viz_kind == "powers":
        states_rot_first = compute_Q_and_diffusions(
            data_rot_first, v_init=v_rot_init_t, t=cfg.t, device=cfg.device, vector_feat_key_for_Q="pos"
        )
        # Also compute the rotated version of the final top-row state for comparison
        v_top_final_rot = rotate_array(states_unrot[-1].numpy(), R90)
    else:
        states_rot_first = compute_Q_and_dyadic_wavelets(
            data_rot_first, v_init=v_rot_init_t, device=cfg.device, J=2, vector_feat_key_for_Q="pos"
        )
        # Rotate the last dyadic band for comparison on the top row
        v_top_final_rot = rotate_array(states_unrot[-1].numpy(), R90)

    # Plot
    if cfg.viz_kind == "powers":
        ncols = cfg.t + 1
    else:
        ncols = 4  # x, (I-Q)x, (Q-Q^2)x, (Q^2-Q^4)x
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8), squeeze=False)

    # Top row: apply operator(s) first, then rotate the final panel
    if cfg.viz_kind == "powers":
        for col in range(ncols):
            ax = axes[0, col]
            if col < cfg.t:
                v_col = states_unrot[col].numpy()
                # title = f"$t={col}$"
                title = rf"$\mathbf{{Q}}^{{{col}}} \mathbf{{w}}$" \
                    if col > 0 else rf"$\mathbf{{w}}$"
                visited_nodes = visited_layers[col] if col < len(visited_layers) else set()
                frontier_nodes = next_step_nodes_from_layers(adj_unrot, visited_layers, col)
                facecolor = None
            else:
                v_col = v_top_final_rot
                # title = f"$t={cfg.t}$, rotated"
                title = rf"$\mathbf{{Q}}^{{{cfg.t}}} \mathbf{{w}}$, rotated"
                visited_nodes = visited_layers[-1] if len(visited_layers) > 0 else set()
                frontier_nodes = next_step_nodes_from_layers(adj_unrot, visited_layers, cfg.t)
                facecolor = cfg.rotate_background_color

            plot_vectors(
                ax,
                xy=xy_np if col < cfg.t else xy_rot_np,
                v=v_col,
                default_color=cfg.default_arrow_color,
                scale=cfg.quiver_scale,
                highlight_center=center_idx,
                frontier_nodes=list(frontier_nodes),
                visited_layer_nodes=list(visited_nodes - {center_idx}),
                center_color=cfg.central_color,
                frontier_color=cfg.neighbor_frontier_color,
                visited_layer_color=cfg.visited_layer_color,
                title=title,
                facecolor=facecolor,
            )
    else:
        dyadic_titles = [
            "$\mathbf{w}$",
            "$(\mathbf{I} - \mathbf{Q})\mathbf{w}$",
            "$(\mathbf{Q} - \mathbf{Q}^{2})\mathbf{w}$",
            "rotate: $(\mathbf{Q}^{2} - \mathbf{Q}^{4})\mathbf{w}$",
        ]
        s_by_col = [0, 1, 2, 4]
        for col in range(ncols):
            ax = axes[0, col]
            if col < ncols - 1:
                v_col = states_unrot[col].numpy()
                title = dyadic_titles[col]
                s = s_by_col[col]
                visited_nodes = visited_layers[s] if s < len(visited_layers) else set()
                frontier_nodes = next_step_nodes_from_layers(adj_unrot, visited_layers, s)
                facecolor = None
                xy_use = xy_np
            else:
                v_col = v_top_final_rot
                title = dyadic_titles[col]
                s = s_by_col[-1]
                visited_nodes = visited_layers[s] if s < len(visited_layers) else set()
                frontier_nodes = next_step_nodes_from_layers(adj_unrot, visited_layers, s)
                facecolor = cfg.rotate_background_color
                xy_use = xy_rot_np

            plot_vectors(
                ax,
                xy=xy_use,
                v=v_col,
                default_color=cfg.default_arrow_color,
                scale=cfg.quiver_scale,
                highlight_center=center_idx,
                frontier_nodes=list(frontier_nodes),
                visited_layer_nodes=list(visited_nodes - {center_idx}),
                center_color=cfg.central_color,
                frontier_color=cfg.neighbor_frontier_color,
                visited_layer_color=cfg.visited_layer_color,
                title=title,
                facecolor=facecolor,
            )

    # Bottom row: rotate first, then diffuse (no extra rotation at the end)
    # For highlighting, recompute center and layers on the rotated graph
    center_idx_rot = find_most_central_node(xy_rot_np)
    if cfg.viz_kind == "powers":
        visited_layers_rot, frontier_layers_rot = build_layer_sets(
            data_rot_first.edge_index, cfg.num_nodes, center_idx_rot, depth=cfg.t
        )
        for col in range(ncols):
            ax = axes[1, col]
            v_col = states_rot_first[col].numpy()
            # title = f"rotate-first, $t={col}$"
            title = rf"rotate-first: $\overline{{\mathbf{{Q}}}}^{{{col}}} \overline{{\mathbf{{w}}}}$" \
                if col > 0 else rf"rotate-first: $\overline{{\mathbf{{w}}}}$"
            visited_nodes = visited_layers_rot[col] if col < len(visited_layers_rot) else set()
            frontier_nodes = next_step_nodes_from_layers(
                build_adjacency_dict(data_rot_first.edge_index, cfg.num_nodes),
                visited_layers_rot,
                col,
            )
            plot_vectors(
                ax,
                xy=xy_rot_np,
                v=v_col,
                default_color=cfg.default_arrow_color,
                scale=cfg.quiver_scale,
                highlight_center=center_idx_rot,
                frontier_nodes=list(frontier_nodes),
                visited_layer_nodes=list(visited_nodes - {center_idx_rot}),
                center_color=cfg.central_color,
                frontier_color=cfg.neighbor_frontier_color,
                visited_layer_color=cfg.visited_layer_color,
                title=title,
                facecolor=None,
            )
    else:
        visited_layers_rot, frontier_layers_rot = build_layer_sets(
            data_rot_first.edge_index, cfg.num_nodes, center_idx_rot, depth=4
        )
        adj_rot = build_adjacency_dict(data_rot_first.edge_index, cfg.num_nodes)
        dyadic_titles_bottom = [
            r"rotate-first: $\overline{\mathbf{w}}$",
            r"rotate-first: $(\mathbf{I} - \overline{\mathbf{Q}})\overline{\mathbf{w}}$",
            r"rotate-first: $(\overline{\mathbf{Q}} - \overline{\mathbf{Q}}^{2})\overline{\mathbf{w}}$",
            r"rotate-first: $(\overline{\mathbf{Q}}^{2} - \overline{\mathbf{Q}}^{4})\overline{\mathbf{w}}$",
        ]
        s_by_col = [0, 1, 2, 4]
        for col in range(ncols):
            ax = axes[1, col]
            v_col = states_rot_first[col].numpy()
            title = dyadic_titles_bottom[col]
            s = s_by_col[col]
            visited_nodes = visited_layers_rot[s] if s < len(visited_layers_rot) else set()
            frontier_nodes = next_step_nodes_from_layers(adj_rot, visited_layers_rot, s)
            plot_vectors(
                ax,
                xy=xy_rot_np,
                v=v_col,
                default_color=cfg.default_arrow_color,
                scale=cfg.quiver_scale,
                highlight_center=center_idx_rot,
                frontier_nodes=list(frontier_nodes),
                visited_layer_nodes=list(visited_nodes - {center_idx_rot}),
                center_color=cfg.central_color,
                frontier_color=cfg.neighbor_frontier_color,
                visited_layer_color=cfg.visited_layer_color,
                title=title,
                facecolor=None,
            )

    fig.suptitle(cfg.plot_title, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_suffix = "dyadic" if cfg.viz_kind == "dyadic" else "powers"
    out_path = f"theory_verification/vector_diffusion_equivariance_viz_{out_suffix}.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")


def parse_args() -> VizConfig:
    parser = argparse.ArgumentParser(description="Vector diffusion equivariance visualization")
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--t", type=int, default=3)
    parser.add_argument("--weighted", action="store_true", help="Use distance-based edge weights")
    parser.add_argument("--gaussian-eps", type=float, default=0.5)
    parser.add_argument("--r-cutoff", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--plot-title", type=str, default=None)
    parser.add_argument(
        "--viz-kind",
        type=str,
        choices=["powers", "dyadic"],
        default="powers",
        help="Visualization mode: 'powers' shows Q^t x sequence; 'dyadic' shows x, (I-Q)x, (Q-Q^2)x, (Q^2-Q^4)x",
    )
    args = parser.parse_args()

    return VizConfig(
        num_nodes=args.num_nodes,
        k=args.k,
        t=args.t,
        weighted=args.weighted,
        gaussian_eps=args.gaussian_eps,
        r_cutoff=args.r_cutoff,
        seed=args.seed,
        device=args.device,
        plot_title=args.plot_title,
        viz_kind=args.viz_kind,
    )


def main() -> None:
    cfg = parse_args()
    visualize(cfg)


if __name__ == "__main__":
    main()


