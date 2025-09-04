"""
Modular SO(d)-equivariant ESCGNN variant with simplified, clean structure.

Class: ESCGNNModular

Key ideas:
- Separate scalar and vector tracks (both ablatable)
- 0th-, 1st-, 2nd-order scattering as in ESCGNN
- Optional BatchNorm for scalars (not for vectors)
- Within-track wavelet mixing via configurable MLPs
  - Scalar: standard MLP across wavelet axis
  - Vector: bias-free linears and scalar gating nonlinearity (equivariant)
- Track mixing: concatenate scalar flattened features with vector invariants
- Node/graph-level heads for scalar and vector targets

This module reuses utilities (e.g., get_Batch_Wjxs) and BaseModule.

--- LaTeX summary ---
\section*{{\modelname} architecture}
Given a graph \(G=(V,E)\) with \(N=|V|\) nodes, optional scalar features \(X_s\in\mathbb{R}^{N\times C}\),
optional vector features \(X_v\in\mathbb{R}^{N\times d}\), and diffusion operators \(P\in \mathbb{R}^{N \times N}\) (scalar) and \(Q \in \mathbb{R}^{Nd \times Nd}\) (vector).

\subsection*{Scattering per track}
\begin{itemize}
\item Scalars: \(W_s^{(0)}=X_s\), \(W_s^{(1)}=\mathcal{W}(X_s;P)\), and second-order interactions
\(W_s^{(2)}=\{\, W_s^{(1)}[\cdots,i] \odot W_s^{(1)}[\cdots,j] \,\}_{i<j}\).
Concatenate along the wavelet axis: \(\tilde W_s=[W_s^{(0)},W_s^{(1)},W_s^{(2)}]\in\mathbb{R}^{N\times C\times W}\). Since scattering coefficients across orders can be of very different magnitudes, optionally apply batch normalization to the scalar scattering coefficients (independently to each channel-wavelet combination).
\item Vectors: similarly apply \(\mathcal{W}(\cdot;Q)\) to \(X_v\) reshaped to \((N \cdot d, 1)\), and reshape result to
\(\tilde W_v\in\mathbb{R}^{N\times 1\times d\times W}\).
\end{itemize}

\subsection*{Within-track wavelet mixing (along the wavelet axis)}
\begin{itemize}
\item Scalars: \(\hat W_s = \mathrm{MLP}_s(\tilde W_s)\), yielding \(\hat W_s\in\mathbb{R}^{N\times C\times W'}\).
\item Vectors (SO(d)-equivariant): bias-free linear layers with scalar gates \(\sigma(\alpha_\ell)\):
\[ y^{(\ell+1)} = \sigma(\alpha_\ell)\, y^{(\ell)} A_\ell, \quad A_\ell \in \mathbb{R}^{W_\ell\times W_{\ell+1}}, \]
applied uniformly across coordinates; output \(\hat W_v\in\mathbb{R}^{N\times 1\times d\times W'}\).
\end{itemize}

\subsection*{Cross-track invariant features}
Form \(t\) by concatenating: (i) flattened scalars \(s=\mathrm{vec}(\hat W_s)\in\mathbb{R}^{N\times (C W')}\)
when present; (ii) vector invariants per wavelet \(n_w=\lVert \hat v_w\rVert_2\) and neighbor cosine statistics
\(\mathrm{cos}_w(u,v)=\tfrac{\langle\hat v_w(u),\hat v_w(v)\rangle}{\lVert\hat v_w(u)\rVert_2\,\lVert\hat v_w(v)\rVert_2}\),
pooled per node via mean/max.

\subsection*{Node-level heads}
\begin{itemize}
\item Scalar targets (or if the vector track is ablated): \(y_s = h_s(t)\in\mathbb{R}^{N\times d_{\text{tar}}}\).
\item Vector targets: gate a weighted sum of vector wavelets. Let \(v_w\in\mathbb{R}^{N\times d}\) be the
\(w\)-th vector wavelet (averaged across the singleton channel). Gates are
\(g=\mathrm{softmax}(h_v(t_{\text{gate}}))\in\Delta^{W'}\) or learned static logits; the prediction is
\[ y_v = \sum_{w=1}^{W'} g_w\, v_w \in \mathbb{R}^{N\times d}. \]
The gate input \(t_{\text{gate}}\) includes vector invariants and (optionally) scalar features.
\end{itemize}

\subsection*{Graph vs. node tasks and final routing}
If the task is graph-level, aggregate node predictions with a permutation-invariant reduce
\(\oplus\in\{\mathrm{sum},\,\mathrm{mean},\,\mathrm{max}\}\) per graph in the batch:
\[ Y^{\text{graph}}_s[b] = \bigoplus_{n\in\mathcal{B}_b} y_s[n], \qquad
   Y^{\text{graph}}_v[b] = \bigoplus_{n\in\mathcal{B}_b} y_v[n]. \]
The module returns node-level outputs for node tasks and aggregated graph-level outputs for graph tasks;
for vector tasks the vector head is used, otherwise the scalar head.

\subsection*{Options} Ablatable: each scalar/vector track, second-order scattering terms can be ablated; the scalar scattering batch normalization layer.
"""

from __future__ import annotations
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from models.base_module import BaseModule
from torch_scatter import scatter
from pyg_utilities import get_Batch_Wjxs


class ScalarWaveletMixMLP(nn.Module):
    """
    Applies an MLP along the last dimension (wavelet axis).

    Input shape: (..., W)
    Output shape: (..., W_out)

    This layer operates independently at each node and channel, viewing the
    wavelet-filtered coefficients as a length-\(W\) vector and learning a
    mapping on that axis only. With no hidden layers, it reduces to a single
    Linear that forms new wavelet coefficients as learned linear combinations
    of the original ones (e.g., \(w'_k = \sum_j a_{kj} w_j\)). With hidden
    layers and nonlinearities, it performs a nonlinear mixing that still acts
    exclusively across the wavelet axis (no mixing across nodes or channels).
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[Sequence[int]] = None,
        nonlin: type[nn.Module] = nn.SiLU,
        dropout_p: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        dims: List[int] = [in_dim] + list(hidden_dims or []) + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            layers.append(nonlin())
            if dropout_p and dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(dims[-2], dims[-1], bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        last_dim = x.shape[-1]
        x_flat = x.reshape(-1, last_dim)
        y = self.net(x_flat)
        return y.reshape(*x.shape[:-1], -1)


class EquivarVectorWaveletMixMLP(nn.Module):
    """
    Bias-free MLP along wavelet axis with scalar gating nonlinearity.

    - Each Linear is bias=False and acts on the wavelet axis only
    - Gating is a scalar SiLU gate per layer (learned parameter), applied
      uniformly to all coordinates, preserving SO(d) equivariance
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[Sequence[int]] = None,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        dims: List[int] = [in_dim] + list(hidden_dims or []) + [out_dim]
        self.linears = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=False) \
            for i in range(len(dims) - 1)
        ])
        # Learned scalar gates (one per layer except last can also be gated)
        self.gates = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0, dtype=torch.float32)) \
            for _ in range(len(self.linears) - 1)
        ])
        self.dropout_p = float(dropout_p) \
            if dropout_p is not None else 0.0

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (..., W)
        last_dim = x.shape[-1]
        y = x.reshape(-1, last_dim)
        for i, lin in enumerate(self.linears):
            y = lin(y)  # learned linear layer
            if i < len(self.linears) - 1:
                gate_val = torch.sigmoid(self.gates[i])  # swish-like scalar gate
                if self.training and self.dropout_p > 0:
                    # Dropout on gate preserves equivariance (uniform over coordinates)
                    gate_val = F.dropout(
                        gate_val, 
                        p=self.dropout_p, 
                        training=True, 
                        inplace=False
                    )
                y = gate_val * y
        return y.reshape(*x.shape[:-1], -1)


class ESCGNNModular(BaseModule):

    MIXING_DEFAULTS = {
        'scalar_hidden_dims': [32, 32],
        'vector_hidden_dims': [32, 32],
        'scalar_dropout_p': 0.0,
        'vector_dropout_p': 0.0,
        'scalar_nonlin': nn.SiLU,
        'W_out_scalar': None,  # if None, keep input W
        'W_out_vector': None,
        'use_scalar_batch_norm': True,
    }

    HEAD_DEFAULTS = {
        'node_scalar_head_hidden': [64, 64],
        'node_scalar_head_nonlin': nn.SiLU,
        'node_scalar_head_dropout': 0.0,
        'vector_gate_hidden': [128, 128],
        'vector_gate_nonlin': nn.SiLU,
        # Gating mode: if True -> Sigmoid + L1 normalize; if False -> Softmax with learned temperature
        'vector_gate_use_sigmoid': True,
        'vector_gate_init_temperature': 1.0,
        # If True, normalize final vector gates; if False, use raw gates
        'normalize_final_vector_gate': True,
        # Vector gating feature controls
        # - If True, include flattened scalar features in vector gate input
        # (if True, this might leak invariant information from the scalar track and
        # break equivariant learning)
        'use_scalar_in_vector_gate': True,
        # - If True, include neighbor cosine stats in vector gate input
        'use_neighbor_cosines': True,
        # - If True, ignore inputs and use learned static per-wavelet weights
        'use_learned_static_vector_weights': False,
    }

    READOUT_DEFAULTS = { # graph-level tasks
        'type': 'mlp',  # 'mlp' (INVARIANT) or 'agg' (EQUIVARIANT)
        'mlp_hidden_dims': [128, 64, 32, 16],
        'mlp_nonlin': 'silu',  # 'silu' or 'relu'
        'node_pool_stats': ['mean', 'max'],  # supports 'mean', 'max', 'sum'
    }

    NEIGHBOR_DEFAULTS = {
        'equal_degree': False,
        'k_neighbors': 5,
        # 'use_padding': True,  # when not equal_degree
        'pool_stats': ['max', 'mean', 'var'],  # supports 'percentiles'|'quantiles', 'min', 'max', 'mean', 'var'
        # NOTE: quantile support is slow
        'quantiles_stride': 0.2,
    }
    
    def __init__(
        self,
        *,
        base_module_kwargs: Dict[str, Any],
        ablate_scalar_track: bool,
        ablate_vector_track: bool,
        ablate_second_order_wavelets: bool = False,
        scalar_track_kwargs: Dict[str, Any],
        vector_track_kwargs: Optional[Dict[str, Any]],
        mixing_kwargs: Optional[Dict[str, Any]] = None,
        neighbor_kwargs: Optional[Dict[str, Any]] = None,
        head_kwargs: Optional[Dict[str, Any]] = None,
        readout_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(**base_module_kwargs)

        # Ablations
        self.ablate_scalar_track = bool(ablate_scalar_track)
        self.ablate_vector_track = bool(ablate_vector_track)

        # Signal to the trainer that this module initializes some submodules lazily
        # (e.g., LazyBatchNorm1d, within-track mixers, heads). The trainer will run
        # a dummy forward pass before DDP wrapping to materialize parameters.
        self.has_lazy_parameter_initialization = True

        # Track kwargs
        self.scalar_track_kwargs = scalar_track_kwargs
        self.vector_track_kwargs = vector_track_kwargs or {}
        self.vector_dim = self.vector_track_kwargs.get('vector_dim', 3)
        self.ablate_second_order_wavelets = bool(ablate_second_order_wavelets)

        # Mixing MLP configuration
        self.mixing_kwargs = {
            **self.MIXING_DEFAULTS, 
            **(mixing_kwargs or {})
        }

        # Neighbor/cosine handling
        self.neighbor_kwargs = {
            **self.NEIGHBOR_DEFAULTS, 
            **(neighbor_kwargs or {})
        }

        # Lazily-created static vector wavelet logits (when configured)
        self.vector_wavelet_logits: Optional[nn.Parameter] = None

        # Scalar BN after scattering (optional)
        self.scalar_bn = None
        if (not self.ablate_scalar_track) and self.mixing_kwargs.get('use_scalar_batch_norm', True):
            # Lazy BN over flattened (C*W_total)
            self.scalar_bn = nn.LazyBatchNorm1d(eps=1e-5, momentum=0.1, affine=True)

        # Placeholders for per-track mixing MLPs; lazy init when W_total known
        self.scalar_mixer: Optional[ScalarWaveletMixMLP] = None
        self.vector_mixer: Optional[EquivarVectorWaveletMixMLP] = None

        # Final heads
        # Node-level scalar head: configurable MLP (bias allowed)
        self.node_scalar_head: Optional[nn.Sequential] = None
        # Node-level vector gating head: produces W' weights for gating sum
        self.node_vector_gate: Optional[nn.Sequential] = None
        # Optional learned temperature for softmax gating (log-parameterized)
        self.vector_gate_log_temperature: Optional[nn.Parameter] = None

        # Graph-level aggregation type
        self.graph_agg: Literal['sum', 'mean', 'max'] = 'sum'

        # Head configuration (can be passed via head_kwargs)
        self.head_kwargs = {
            **self.HEAD_DEFAULTS, 
            **(head_kwargs or {})
        }

        # Readout configuration (graph-level)
        self.readout_kwargs = {
            **self.READOUT_DEFAULTS,
            **(readout_kwargs or {})
        }

        # Graph-level readout MLPs (lazy)
        self.graph_scalar_readout_mlp: Optional[nn.Sequential] = None
        # Note: For vector graph readouts, we keep aggregation-only to preserve equivariance

        # Cache for upper-triangular pair indices by wavelet width (nW)
        # Stored on CPU and moved to the active device on demand
        self._triu_indices_cache: Dict[int, Tuple[Tensor, Tensor]] = {}

    # ------------------------- Helpers -------------------------
    def _scatter(
        self,
        *,
        track: Literal['scalar', 'vector'],
        x0: Tensor,
        P_or_Q: Tensor,
        kwargs: Dict[str, Any],
    ) -> Tensor:
        """
        Concatenate 0th, 1st, and 2nd order scattering outputs along wavelet axis.
        - Scalar input x0 shape: (N, C) -> returns (N, C, W)
        - Vector input x0 shape: (N, d) -> returns (N, 1, d, W)
        """
        # print(f'_scatter: track={track}')

        if track == 'scalar':
            # First-order scattering
            W1 = get_Batch_Wjxs(
                x=x0,
                P_sparse=P_or_Q,
                # vector_dim=None, # deprecated
                **kwargs['diffusion_kwargs'],
            )  # (N, C, W1)
            # Optional second-order interactions via indexed pairs
            W2_list = []
            if not self.ablate_second_order_wavelets:
                nW = W1.shape[-1]
                i_idx, j_idx = self._get_triu_pair_indices(nW, device=W1.device)
                W2 = W1[..., i_idx] * W1[..., j_idx]  # (N, C, nW*(nW-1)/2)
                W2_list = [W2]
            # Zeroth order
            W0 = x0.unsqueeze(-1)  # (N, C, 1)
            return torch.cat([W0, W1] + W2_list, dim=-1)  # (N, C, W)

        # Vector track
        N, d = x0.shape
        flat = x0.reshape(N * d, 1)
        W1v = get_Batch_Wjxs(
            x=flat,
            P_sparse=P_or_Q,
            # vector_dim=d, # deprecated
            **kwargs['diffusion_kwargs'],
        )  # (N*d, 1, W1)
        nW = W1v.shape[-1]
        # Optional second-order interactions via indexed pairs
        W2v_list = []
        if not self.ablate_second_order_wavelets:
            i_idx, j_idx = self._get_triu_pair_indices(nW, device=W1v.device)
            W2v = W1v[..., i_idx] * W1v[..., j_idx]  # (N*d, 1, nW*(nW-1)/2)
            W2v_list = [W2v]
        W0v = flat.unsqueeze(-1)  # (N*d, 1, 1)
        W_tot = torch.cat([W0v, W1v] + W2v_list, dim=-1)  # (N*d, 1, W)
        W_tot = W_tot.view(N, d, 1, -1)
        return W_tot.permute(0, 2, 1, 3)  # (N, 1, d, W)

    def _lazy_init_within_track_mixers(
        self,
        W_scalar: Optional[Tensor],
        W_vector: Optional[Tensor],
    ) -> None:
        # Determine in/out wavelet dims
        if (self.scalar_mixer is None) and (W_scalar is not None):
            W_in = W_scalar.shape[-1]
            W_out = self.mixing_kwargs.get('W_out_scalar') or W_in
            self.scalar_mixer = ScalarWaveletMixMLP(
                in_dim=W_in,
                out_dim=W_out,
                hidden_dims=self.mixing_kwargs.get('scalar_hidden_dims'),
                nonlin=self.mixing_kwargs.get('scalar_nonlin', nn.SiLU),
                dropout_p=self.mixing_kwargs.get('scalar_dropout_p', 0.0),
                bias=True,
            ).to(W_scalar.device)

        if (self.vector_mixer is None) and (W_vector is not None):
            W_in = W_vector.shape[-1]
            W_out = self.mixing_kwargs.get('W_out_vector') or W_in
            self.vector_mixer = EquivarVectorWaveletMixMLP(
                in_dim=W_in,
                out_dim=W_out,
                hidden_dims=self.mixing_kwargs.get('vector_hidden_dims'),
                dropout_p=self.mixing_kwargs.get('vector_dropout_p', 0.0),
            ).to(W_vector.device)

    def _optional_scalar_bn(
        self,
        W_scalar: Tensor,
    ) -> Tensor:
        if self.scalar_bn is None:
            return W_scalar
        N, C, W = W_scalar.shape
        flat = W_scalar.reshape(N, C * W)
        flat = self.scalar_bn(flat)
        return flat.reshape(N, C, W)

    def _get_vector_invariants(
        self,
        Wv: Tensor,
        batch: Batch,
    ) -> Tensor:  # type: ignore[name-defined]
        """
        Compute vector invariants per node for track-mixing:
        - Norm per wavelet of mean across coordinates: (N, W')
        - Neighbor cosine similarities per wavelet: (N, W' * m) pooled stats
        Returns shape: (N, W' + W' * m)
        """
        # Wv shape: (N, W', 1, d)
        N, Wp = Wv.shape[0], Wv.shape[1]
        v_mean_per_wavelet = Wv.squeeze(2)  # (N, W', d)
        norms = torch.linalg.norm(v_mean_per_wavelet, dim=-1)  # (N, W')

        edge_index = batch.edge_index  # (2, E)
        src, dst = edge_index[0], edge_index[1]

        # Cosines per wavelet between node vs neighbor vectors
        # Broadcast across wavelets; compute per-edge per-wavelet cosine
        v = v_mean_per_wavelet  # (N, W', d)
        v_norm = torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=1e-8)  # (N, W', 1)
        v_dst = v[dst]  # (E, W', d)
        v_src = v[src]  # (E, W', d)
        denom = (v_norm[dst].squeeze(-1) * v_norm[src].squeeze(-1))  # (E, W')
        cos_all = (v_dst * v_src).sum(dim=-1) / denom  # (E, W')

        stats: Sequence[str] = self.neighbor_kwargs.get(
            'pool_stats', 
            ['percentiles', 'min', 'max', 'mean']
        )
        pooled_per_wavelet: List[Tensor] = []
        # Compute mean once if requested directly or needed for variance
        pooled_mean: Optional[Tensor] = None
        if ('mean' in stats) or ('var' in stats):
            pooled_mean = scatter(cos_all, dst, dim=0, dim_size=N, reduce='mean')  # (N, W')
            if 'mean' in stats:
                pooled_per_wavelet.append(pooled_mean)
        if 'max' in stats:
            pooled_per_wavelet.append(scatter(cos_all, dst, dim=0, dim_size=N, reduce='max'))   # (N, W')
        if 'min' in stats:
            pooled_per_wavelet.append(scatter(cos_all, dst, dim=0, dim_size=N, reduce='min'))   # (N, W')
        if 'var' in stats:
            # Efficient variance via E[x^2] - (E[x])^2
            mean_of_squares = scatter(cos_all * cos_all, dst, dim=0, dim_size=N, reduce='mean')  # (N, W')
            # pooled_mean is guaranteed computed above when 'var' in stats
            var_vals = torch.clamp(mean_of_squares - pooled_mean.pow(2), min=0.0)
            pooled_per_wavelet.append(var_vals)

        # Percentile/quantile pooling (treated as multiple stats)
        # Strategy overview:
        # - We want per-node, per-wavelet quantiles of neighbor cosine similarities.
        # - We first sort edges by destination node index so each node's edges form a
        #   contiguous segment. This lets us index a node's subarray via prefix sums.
        # - Given a stride s in (0,1), we form quantile levels {s, 2s, 3s, ...} < 1.0.
        # - For each level p and each node n with degree deg_n > 0, we select the k-th
        #   smallest value with k = floor((deg_n - 1) * p) for each wavelet column.
        #   This corresponds to a "lower"-interpolation quantile. We use torch.kthvalue
        #   along dim=0 to compute all wavelet columns at once for that node's segment.
        # - Nodes with zero degree contribute zeros for that quantile (consistent with
        #   the surrounding zero-initialization used elsewhere).
        # Notes:
        # - Complexity is O(E log E) for the sort once, then O(E + N * Q * log deg)
        #   naively; here kthvalue is O(deg) without a full sort per node per level.
        # - This could be vectorized further, but the current approach is simple and
        #   clear while remaining efficient for typical graph sizes.
        use_quantiles = ('quantiles' in stats) or ('percentiles' in stats)
        if use_quantiles:
            stride = float(self.neighbor_kwargs.get('quantiles_stride', 0.2))
            # Construct levels in (0,1), e.g., stride=0.2 -> [0.2, 0.4, 0.6, 0.8]
            if stride > 0.0 and stride < 1.0:
                levels = torch.arange(stride, 1.0, stride, device=Wv.device)
                if levels.numel() > 0:
                    # 1) Group edges by destination to form contiguous segments per node
                    perm = torch.argsort(dst, stable=True)
                    dst_sorted = dst[perm]
                    cos_sorted = cos_all[perm]  # (E, W')
                    counts = torch.bincount(dst_sorted, minlength=N)
                    # Prefix sum of counts gives the starting index of each node's segment
                    starts = torch.cumsum(counts, dim=0) - counts  # (N,)
                    for p in levels.tolist():
                        # 2) Allocate output for this quantile level across all nodes
                        out_p = torch.zeros((N, Wp), device=Wv.device, dtype=cos_all.dtype)
                        for n in range(N):
                            cnt = int(counts[n].item())
                            if cnt <= 0:
                                continue
                            # 3) Node n's segment across all edges, for all wavelets
                            seg = cos_sorted[starts[n]: starts[n] + cnt, :]  # (cnt, W')
                            # Choose k = floor((deg-1) * p) to implement lower quantile
                            k = int((cnt - 1) * p)
                            # kthvalue with dim=0 returns per-column k-th smallest (1-indexed)
                            out_p[n] = torch.kthvalue(seg, k=k + 1, dim=0).values
                        pooled_per_wavelet.append(out_p)

        if pooled_per_wavelet:
            pooled = torch.cat(pooled_per_wavelet, dim=1)  # (N, W' * m)
        else:
            pooled = torch.zeros(N, 0, device=Wv.device)

        return torch.cat([norms, pooled], dim=1)  # (N, W' + W'*m)

    # --------------------------- 2nd-order wavelets helper ---------------------------
    def _get_triu_pair_indices(self, nW: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """
        Return cached (i,j) indices for the strictly upper-triangular pairs of an nW x nW matrix.
        Indices are cached on CPU and moved to the requested device.
        """
        if nW not in self._triu_indices_cache:
            ij = torch.triu_indices(nW, nW, offset=1, device='cpu')  # (2, K)
            self._triu_indices_cache[nW] = (ij[0].contiguous(), ij[1].contiguous())
        i_cpu, j_cpu = self._triu_indices_cache[nW]
        return i_cpu.to(device), j_cpu.to(device)

    # --------------------------- Node-level vector output helper ---------------------------
    def _compute_node_vector_output(
        self,
        *,
        W_vector: Tensor,               # (N, W', 1, d)
        W_scalar: Optional[Tensor],     # (N, W', C) or None
        vector_invariants: Optional[Tensor],
        batch: Batch,                   # for neighbor info if needed
        device: torch.device,
    ) -> Tensor:
        """
        Compute node-level vector predictions by per-wavelet gating and summation.

        Strategy:
        - Build per-node gate features t_gate by concatenating, when enabled:
          (a) flattened scalar track (if present), (b) vector norms per wavelet, and
          (c) optional neighbor cosine statistics.
        - Two gating modes are supported:
          1) Static per-wavelet weights: a single learnable logit per wavelet shared across nodes.
          2) Data-driven gates: an MLP maps t_gate to per-node logits.
        - Normalization behavior is controlled by head_kwargs['normalize_final_vector_gate']:
          - If True and using sigmoid gates: apply sigmoid then L1 normalization across wavelets.
          - If True and using softmax mode: apply temperature-softmax across wavelets.
          - If False: use raw sigmoid outputs (sigmoid mode) or raw logits (softmax mode) without normalization.
        - The final node vector is the weighted sum across wavelets of W_vector, preserving SO(d) equivariance.

        Returns:
        - Tensor of shape (N, 1, d) containing node-level vector predictions.
        """
        # Number of mixed wavelets
        Wp = W_vector.shape[1]

        # Assemble gate features
        gate_feats: List[Tensor] = []
        if W_scalar is not None:
            Ns, Wps, C = W_scalar.shape
            gate_feats.append(W_scalar.reshape(Ns, Wps * C))
        if vector_invariants is None:
            vector_invariants = self._get_vector_invariants(W_vector, batch)
        gate_feats.append(vector_invariants[:, :Wp])  # norms per wavelet
        if self.head_kwargs.get('use_neighbor_cosines'):
            gate_feats.append(vector_invariants[:, Wp:])
        t_gate = torch.cat(gate_feats, dim=1) if gate_feats else None # (N, W' + W'*m),
        # where m is the number of neighbor cosine statistics.

        normalize_final: bool = bool(self.head_kwargs.get('normalize_final_vector_gate', True))
        use_sigmoid_gate: bool = bool(self.head_kwargs.get('vector_gate_use_sigmoid'))

        # Case 1: static per-wavelet weights shared across nodes
        if self.head_kwargs.get('use_learned_static_vector_weights'):
            if (self.vector_wavelet_logits is None) or (self.vector_wavelet_logits.numel() != Wp):
                self.vector_wavelet_logits = nn.Parameter(torch.zeros(Wp, device=device))
            if normalize_final:
                gates = torch.softmax(self.vector_wavelet_logits, dim=0).unsqueeze(0).expand(W_vector.size(0), -1)
            else:
                gates = self.vector_wavelet_logits.unsqueeze(0).expand(W_vector.size(0), -1)

        # Case 2: data-driven gates from t_gate via MLP
        elif t_gate is not None:
            need_reinit_gate = False
            expected_in_dim = t_gate.shape[1]
            if self.node_vector_gate is None:
                need_reinit_gate = True
            else:
                try:
                    first_linear = next(m for m in self.node_vector_gate if isinstance(m, nn.Linear))
                    last_linear = next(m for m in self.node_vector_gate[::-1] if isinstance(m, nn.Linear))
                    if last_linear.out_features != Wp or first_linear.in_features != expected_in_dim:
                        need_reinit_gate = True
                except StopIteration:
                    need_reinit_gate = True
            if need_reinit_gate:
                self._lazy_init_node_vector_gate(in_dim=expected_in_dim, out_dim=Wp, device=device)
            logits = self.node_vector_gate(t_gate)  # (N, W')
            if use_sigmoid_gate:
                gates = torch.sigmoid(logits)
                if normalize_final:
                    # L1 normalization
                    gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)
            else:
                if normalize_final:
                    if self.vector_gate_log_temperature is None:
                        init_temp = float(self.head_kwargs.get('vector_gate_init_temperature', 1.0))
                        init_temp = max(init_temp, 1e-3)
                        self.vector_gate_log_temperature = nn.Parameter(
                            torch.log(torch.tensor(init_temp, dtype=torch.float32, device=logits.device))
                        )
                    temperature = torch.exp(self.vector_gate_log_temperature)
                    gates = torch.softmax(logits / temperature.clamp_min(1e-3), dim=1)
                else:
                    gates = logits

        # Case 3: uniform fallback
        else:
            gates = torch.full((W_vector.size(0), Wp), 1.0 / max(Wp, 1), device=device)

        # Weighted sum across wavelets -> (N, 1, d)
        vec = (W_vector.squeeze(2) * gates.unsqueeze(-1)).sum(dim=1)
        return vec.unsqueeze(1)

    def _infer_device_from_batch(self, batch: Batch) -> torch.device:  # type: ignore[name-defined]
        """
        Infer device from available tensors in the incoming Batch. Avoids
        relying on existing model parameters since this module is lazily
        initialized and may have zero parameters before the first forward.
        """
        # Prefer scalar feature tensor
        if not self.ablate_scalar_track and hasattr(batch, self.scalar_track_kwargs['feature_key']):
            _x = getattr(batch, self.scalar_track_kwargs['feature_key'])
            if isinstance(_x, torch.Tensor):
                return _x.device
        # Fallback to vector feature tensor
        if not self.ablate_vector_track and hasattr(batch, self.vector_track_kwargs['feature_key']):
            _xv = getattr(batch, self.vector_track_kwargs['feature_key'])
            if isinstance(_xv, torch.Tensor):
                return _xv.device
        # Default CPU
        return torch.device('cpu')

    # --------------------------- Forward ---------------------------
    def forward(
        self, batch
    ) -> Dict[str, Tensor]:  # type: ignore[override]
        
        # Initialize outputs dict
        outputs: Dict[str, Tensor] = {}

        # Extract base tensors
        device = self._infer_device_from_batch(batch)
        x_s = None if self.ablate_scalar_track \
            else getattr(batch, self.scalar_track_kwargs['feature_key'])
        x_v = None if self.ablate_vector_track \
            else getattr(batch, self.vector_track_kwargs['feature_key'])
        P = getattr(batch, self.scalar_track_kwargs['diffusion_op_key']) \
            if not self.ablate_scalar_track else None
        Q = getattr(batch, self.vector_track_kwargs['diffusion_op_key']) \
            if not self.ablate_vector_track else None

        if x_s is not None:
            x_s = x_s.to(device)
            P = P.to(device)
        if x_v is not None:
            x_v = x_v.to(device)
            Q = Q.to(device)

        # Task flag used to decide which tracks are needed
        task = self.task

        # --- 1. Scattering ---
        # 0th, 1st, 2nd order scattering coeffs, concatenated
        W_scalar = None
        
        # Always compute scalar track when not ablated so it can participate in vector tasks
        if (not self.ablate_scalar_track) \
        and (x_s is not None) \
        and (P is not None):
            W_scalar = self._scatter(
                track='scalar', 
                x0=x_s, 
                P_or_Q=P, 
                kwargs=self.scalar_track_kwargs,
            )  # (N, C, W)
            # Optional BN over flattened features
            W_scalar = self._optional_scalar_bn(W_scalar)

        W_vector = None
        if not self.ablate_vector_track:
            W_vector = self._scatter(
                track='vector', 
                x0=x_v, 
                P_or_Q=Q, 
                kwargs=self.vector_track_kwargs,
            )  # (N, 1, d, W)

        # --- 2. Within-track mixing ---
        # Initialize mixers lazily now that W dims are known
        vec_for_init = W_vector.permute(0, 2, 1, 3) \
            if (W_vector is not None) else None  # (..., W)
        self._lazy_init_within_track_mixers(W_scalar, vec_for_init)

        # Apply within-track wavelet mixing MLPs
        if W_scalar is not None and self.scalar_mixer is not None:
            # (N, C, W) -> (N, C, W') then permute to (N, W', C)
            W_scalar = self.scalar_mixer(W_scalar)
            W_scalar = W_scalar.permute(0, 2, 1)

        if W_vector is not None and self.vector_mixer is not None:
            # Rearrange to (N, d, 1, W) -> apply mixer on last dim -> (N, d, 1, W')
            Wv = W_vector.permute(0, 2, 1, 3)
            # (vector track mixer MLP has no bias in layers)
            Wv = self.vector_mixer(Wv)
            # Return to (N, W', 1, d)
            W_vector = Wv.permute(0, 3, 2, 1)

        # --- 3. Cross-track mixing ---
        # Gather invariant features from both tracks (use norms and cosines from vector track)
        t_list: List[Tensor] = []
        vector_invariants: Optional[Tensor] = None
        if W_scalar is not None:
            Ns, Wp, C = W_scalar.shape
            t_list.append(W_scalar.reshape(Ns, Wp * C))
        if W_vector is not None:
            vector_invariants = self._get_vector_invariants(W_vector, batch)
            t_list.append(vector_invariants)
        t = None
        if t_list:
            t = torch.cat(t_list, dim=1)  # (N, W' + W'*m)

        # Short-circuit, if graph-level scalar target task: MLP readout directly from pooled t (assumes batch.batch is present)
        if ('graph' in task) and (('vector' not in task) or self.ablate_vector_track):
            if str(self.readout_kwargs.get('type', 'mlp')).lower() == 'mlp' \
            and (t is not None) and hasattr(batch, 'batch'):
                outputs['graph_scalar'] = self._graph_scalar_readout_from_pooled_t(t=t, batch_index=batch.batch)
                outputs['preds'] = outputs['graph_scalar']
                return outputs

        # --- 4. Node-level predictions ---
        # (always computed, and pooled / read out after if needed)
        # a. Scalar targets or ablated vector track
        if ('vector' not in task) or self.ablate_vector_track:
            # Scalar target (node-level): MLP head from t -> (N, d_target)
            if self.node_scalar_head is None:
                in_dim = (t.shape[1] if t is not None else 0)
                self._lazy_init_node_scalar_head(in_dim=in_dim, out_dim=self.target_dim, device=device)
            pred = self.node_scalar_head(t)
            # For node-level scalar targets with no vector track -> prediction is a MLP head from t -> (N, d_target)
            outputs['node_scalar'] = pred

        # b. Vector targets (and not-ablated vector track)
        # We use a vector gating MLP to gate the sum of the wavelets
        # (when not ablating the vector track)
        else:
            # Vector target (node-level): delegate to helper
            outputs['node_vector'] = self._compute_node_vector_output(
                W_vector=W_vector,
                W_scalar=W_scalar,
                vector_invariants=vector_invariants,
                batch=batch,
                device=device,
            )

        # --- 5. [Optional] Graph-level readout heads ---
        # (Simple aggregations of node-level features)
        if 'graph' in task:
            agg = getattr(self, 'graph_agg', 'sum')
            batch_index = batch.batch if hasattr(batch, 'batch') else None
            # Graph-level vector target: aggregate node vector predictions
            if 'vector' in task and 'node_vector' in outputs:
                v = outputs['node_vector'].squeeze(1)  # (N, d)
                out_vec = self._aggregate_nodes(v, batch_index=batch_index, agg=agg)
                outputs['graph_vector'] = out_vec # (B, d)
                
            if 'node_scalar' in outputs:
                s = outputs['node_scalar']  # (N, d_target)
                out_s = self._aggregate_nodes(s, batch_index=batch_index, agg=agg)  # (B, d_target)
                readout_type = str(self.readout_kwargs.get('type', 'mlp')).lower()
                if readout_type == 'mlp':
                    # Lazily build MLP and apply
                    if self.graph_scalar_readout_mlp is None:
                        self._lazy_init_graph_scalar_readout_mlp(
                            in_dim=out_s.shape[1],
                            out_dim=self.target_dim,
                            device=out_s.device,
                        )
                    outputs['graph_scalar'] = self.graph_scalar_readout_mlp(out_s)
                else:
                    outputs['graph_scalar'] = out_s # (B, d_target)

        # --- 6. Select final predictions ---
        # Decide which tensor is the final prediction depending on task/target
        if 'graph' in task:
            if 'vector' in task and ('graph_vector' in outputs):
                outputs['preds'] = outputs['graph_vector']
            elif ('graph_scalar' in outputs):
                outputs['preds'] = outputs['graph_scalar']
        elif 'node' in task:
            if 'vector' in task and ('node_vector' in outputs):
                outputs['preds'] = outputs['node_vector']
            elif ('node_scalar' in outputs):
                outputs['preds'] = outputs['node_scalar']

        return outputs

    # -------------------- Epoch-zero initializer --------------------
    def run_epoch_zero_methods(self, batch: Batch) -> None:  # type: ignore[name-defined]
        """
        Materialize lazily-created submodules (mixers/heads) without requiring
        a full training step. This is invoked by the trainer at epoch 0 and is
        also safe to call during a pre-DDP dummy pass.
        """
        device = self._infer_device_from_batch(batch)
        x_s = None if self.ablate_scalar_track \
            else getattr(batch, self.scalar_track_kwargs['feature_key'])
        x_v = None if self.ablate_vector_track \
            else getattr(batch, self.vector_track_kwargs['feature_key'])
        P = getattr(batch, self.scalar_track_kwargs['diffusion_op_key']) \
            if not self.ablate_scalar_track else None
        Q = getattr(batch, self.vector_track_kwargs['diffusion_op_key']) \
            if not self.ablate_vector_track else None

        if x_s is not None:
            x_s = x_s.to(device)
            P = P.to(device)
        if x_v is not None:
            x_v = x_v.to(device)
            Q = Q.to(device)

        # Compute minimal scattering tensors to determine mixer/head shapes
        W_scalar = None
        if not self.ablate_scalar_track and x_s is not None and P is not None:
            W_scalar = self._scatter(
                track='scalar', x0=x_s, P_or_Q=P, kwargs=self.scalar_track_kwargs
            )
            W_scalar = self._optional_scalar_bn(W_scalar)

        W_vector = None
        if not self.ablate_vector_track and x_v is not None and Q is not None:
            W_vector = self._scatter(
                track='vector', x0=x_v, P_or_Q=Q, kwargs=self.vector_track_kwargs
            )

        # Initialize mixers based on discovered W-dimensions
        vec_for_init = W_vector.permute(0, 2, 1, 3) if (W_vector is not None) else None
        self._lazy_init_within_track_mixers(W_scalar, vec_for_init)

        # Apply within-track mixers identically to forward()
        if W_scalar is not None and self.scalar_mixer is not None:
            W_scalar = self.scalar_mixer(W_scalar)
            W_scalar = W_scalar.permute(0, 2, 1)  # (N, W', C)
        if W_vector is not None and self.vector_mixer is not None:
            Wv = W_vector.permute(0, 2, 1, 3)      # (N, d, 1, W)
            Wv = self.vector_mixer(Wv)
            W_vector = Wv.permute(0, 3, 2, 1)      # (N, W', 1, d)

        # Build cross-track feature t identically to forward()
        t_list: List[Tensor] = []
        vector_invariants: Optional[Tensor] = None
        if W_scalar is not None:
            Ns, Wp, C = W_scalar.shape
            t_list.append(W_scalar.reshape(Ns, Wp * C))
        if W_vector is not None:
            vector_invariants = self._get_vector_invariants(W_vector, batch)
            t_list.append(vector_invariants)
        t = torch.cat(t_list, dim=1) if len(t_list) > 0 else None

        # Initialize heads according to task routing
        task = self.task
        if ('vector' not in task) or self.ablate_vector_track:
            # For graph-scalar tasks with MLP readout, skip node head init to avoid unused params
            use_graph_scalar_mlp = (
                ('graph' in task)
                and (str(self.readout_kwargs.get('type', 'mlp')).lower() == 'mlp')
            )
            if not use_graph_scalar_mlp:
                # Node scalar head for scalar tasks
                in_dim = (t.shape[1] if t is not None else 0)
                if self.node_scalar_head is None:
                    self._lazy_init_node_scalar_head(in_dim=in_dim, out_dim=self.target_dim, device=device)
        else:
            # Vector gate head for vector tasks
            if W_vector is not None:
                Wp = W_vector.shape[1]
                # Build gate input exactly as in forward()
                gate_feats: List[Tensor] = []
                # Always include flattened scalar features when scalar track is enabled
                if W_scalar is not None:
                    Ns, Wps, C = W_scalar.shape
                    gate_feats.append(W_scalar.reshape(Ns, Wps * C))
                # Reuse precomputed vector invariants; compute if missing
                if vector_invariants is None:
                    vector_invariants = self._get_vector_invariants(W_vector, batch)
                # Norms per wavelet (first Wp cols)
                gate_feats.append(vector_invariants[:, :Wp])  # (N, W')
                # Optional neighbor cosine pooled stats (remaining cols)
                if self.head_kwargs.get('use_neighbor_cosines'):
                    gate_feats.append(vector_invariants[:, Wp:])
                t_gate = torch.cat(gate_feats, dim=1) if gate_feats else None

                if self.node_vector_gate is None:
                    in_dim = (t_gate.shape[1] if t_gate is not None else 0)
                    self._lazy_init_node_vector_gate(in_dim=in_dim, out_dim=Wp, device=device)

        # Initialize graph-level scalar readout MLP if requested and task is graph-scalar
        if ('graph' in task) and (('vector' not in task) or self.ablate_vector_track):
            if str(self.readout_kwargs.get('type', 'mlp')).lower() == 'mlp':
                # Build pooled t to determine correct input dim (assumes batch.batch present)
                if hasattr(batch, 'batch'):
                    t_list: List[Tensor] = []
                    if W_scalar is not None:
                        Ns, Wp, C = W_scalar.shape
                        t_list.append(W_scalar.reshape(Ns, Wp * C))
                    if W_vector is not None:
                        # Reuse precomputed invariants; compute if missing
                        if vector_invariants is None:
                            vector_invariants = self._get_vector_invariants(W_vector, batch)
                        t_list.append(vector_invariants)
                    t_epoch0 = torch.cat(t_list, dim=1) if len(t_list) > 0 else None
                    if t_epoch0 is not None:
                        _ = self._graph_scalar_readout_from_pooled_t(t=t_epoch0, batch_index=batch.batch)

    # --------------------------- Lazy init helpers ---------------------------
    def _lazy_init_node_scalar_head(
        self, 
        *, 
        in_dim: int, 
        out_dim: int, 
        device: torch.device,
    ) -> None:
        hidden: List[int] = list(self.head_kwargs.get('node_scalar_head_hidden'))
        nonlin_cls: type[nn.Module] = self.head_kwargs.get('node_scalar_head_nonlin', nn.SiLU)
        dropout_p: float = float(self.head_kwargs.get('node_scalar_head_dropout', 0.0))
        dims = [in_dim] + hidden + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            layers.append(nonlin_cls())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(dims[-2], dims[-1], bias=True))
        self.node_scalar_head = nn.Sequential(*layers).to(device)

    def _lazy_init_node_vector_gate(
        self,
        *, 
        in_dim: int, 
        out_dim: int, 
        device: torch.device,
    ) -> None:
        hidden: List[int] = list(self.head_kwargs.get('vector_gate_hidden'))
        nonlin_cls: type[nn.Module] = self.head_kwargs.get('vector_gate_nonlin', nn.SiLU)
        dims = [in_dim] + hidden + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            layers.append(nonlin_cls())
        layers.append(nn.Linear(dims[-2], dims[-1], bias=True))
        self.node_vector_gate = nn.Sequential(*layers).to(device)

        # Initialize learned temperature parameter for softmax gating when requested
        use_sigmoid: bool = bool(self.head_kwargs.get('vector_gate_use_sigmoid', True))
        if not use_sigmoid:
            init_temp = float(self.head_kwargs.get('vector_gate_init_temperature', 1.0))
            init_temp = max(init_temp, 1e-3)
            if (self.vector_gate_log_temperature is None) or (not isinstance(self.vector_gate_log_temperature, nn.Parameter)):
                self.vector_gate_log_temperature = nn.Parameter(
                    torch.log(torch.tensor(init_temp, dtype=torch.float32, device=device))
                )
            else:
                # Ensure it's on the correct device
                self.vector_gate_log_temperature = nn.Parameter(self.vector_gate_log_temperature.detach().to(device))

    def _lazy_init_graph_scalar_readout_mlp(
        self,
        *,
        in_dim: int,
        out_dim: int,
        device: torch.device,
    ) -> None:
        hidden: List[int] = list(self.readout_kwargs.get('mlp_hidden_dims', [128, 64, 32, 16]))
        nonlin_name: str = str(self.readout_kwargs.get('mlp_nonlin', 'silu')).lower()
        nonlin_cls: type[nn.Module] = nn.SiLU if nonlin_name == 'silu' else nn.ReLU
        dims = [in_dim] + hidden + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            layers.append(nonlin_cls())
        layers.append(nn.Linear(dims[-2], dims[-1], bias=True))
        self.graph_scalar_readout_mlp = nn.Sequential(*layers).to(device)

    def _graph_scalar_readout_from_pooled_t(
        self,
        *,
        t: Tensor,                  # (N, F)
        batch_index: Tensor,        # (N,)
    ) -> Tensor:                    # returns (B, target_dim)
        batch_index = batch_index.to(device=t.device, dtype=torch.long)
        B = int(batch_index.max().item()) + 1
        stats: Sequence[str] = self.readout_kwargs.get('node_pool_stats', ['mean', 'max'])
        pooled_stats: List[Tensor] = []
        if 'mean' in stats:
            pooled_stats.append(scatter(t, batch_index, dim=0, dim_size=B, reduce='mean'))
        if 'max' in stats:
            pooled_stats.append(scatter(t, batch_index, dim=0, dim_size=B, reduce='max'))
        if 'sum' in stats:
            pooled_stats.append(scatter(t, batch_index, dim=0, dim_size=B, reduce='sum'))
        t_agg = torch.cat(pooled_stats, dim=1) if pooled_stats else scatter(t, batch_index, dim=0, dim_size=B, reduce='mean')
        if self.graph_scalar_readout_mlp is None:
            self._lazy_init_graph_scalar_readout_mlp(
                in_dim=t_agg.shape[1],
                out_dim=self.target_dim,
                device=t_agg.device,
            )
        return self.graph_scalar_readout_mlp(t_agg)

    # --------------------------- Aggregation helper ---------------------------
    def _aggregate_nodes(
        self,
        x: Tensor,
        *,
        batch_index: Optional[Tensor],
        agg: Literal['sum', 'mean', 'max'] = 'sum',
    ) -> Tensor:
        if batch_index is None:
            return x.unsqueeze(0)
        batch_index = batch_index.to(device=x.device, dtype=torch.long)
        B = int(batch_index.max().item()) + 1
        out = scatter(
            src=x, 
            index=batch_index,
            dim=0, 
            dim_size=B, 
            reduce=agg,
        )
        return out


