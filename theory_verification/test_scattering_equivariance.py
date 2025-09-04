#!/usr/bin/env python3

"""
Quick equivariance residual test for node-level vector targets.

Usage example:
  python3.11 scripts/python/test_equivariance.py \
    --dataset=ellipsoids \
    --config=config/yaml_files/ellipsoid_escgnn_modular_minimal.yaml

This script loads the config/dataset/model similarly to main_training.py,
fetches one batch, applies a random 3x3 rotation R to the vector features,
runs the model on original and rotated batches, and reports the residual:
  || pred(R·x) - R·pred(x) || / || R·pred(x) ||

It is intended for node-level vector tasks. For other tasks it will exit.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from config.arg_parsing import get_clargs
from config.config_manager import ConfigManager
from training.prep_dataset import load_dataset, create_dataloaders
from training.prep_model import prepare_escgnn_model
from data_processing.process_pyg_data import (
    process_pyg_data,
    get_C_i_dict,
    get_l_singular_vecs,
    get_local_pca_kernel_weights,
    match_B_col_directions_to_A,
    calc_O_ij,
)


def random_rotation_matrix_3d(device: torch.device) -> torch.Tensor:
    """Generate a random 3D rotation matrix using QR decomposition."""
    A = torch.randn(3, 3, device=device)
    # QR gives orthonormal Q; adjust sign to ensure det=+1
    Q, R = torch.linalg.qr(A)
    d = torch.sign(torch.linalg.det(Q))
    Q[:, -1] = Q[:, -1] * d
    return Q


@torch.no_grad()
def apply_rotation_to_batch(batch, R: torch.Tensor, keys: list[str]) -> None:
    """
    In-place rotate specified 3D vector attributes in a PyG Batch/Data.
    Shape expected: (N, 3) for each key.
    """
    for k in keys:
        if hasattr(batch, k):
            v = getattr(batch, k)
            if isinstance(v, torch.Tensor) and v.dim() == 2 and v.size(-1) == 3:
                # x @ R^T (row vectors)
                setattr(batch, k, v @ R.T)


@torch.no_grad()
def compute_equivariance_residual(
    model,
    batch,
    vector_key: str,
    device: torch.device,
    cfg=None,
    R: torch.Tensor = None,
) -> dict:
    model.eval()

    # Ensure batch is on device
    batch = batch.to(device)

    # Run epoch 0 method
    if hasattr(model, 'run_epoch_zero_methods'):
        model.run_epoch_zero_methods(batch)

    # Baseline prediction
    out0 = model(batch)
    preds0 = out0.get('preds', None)
    if preds0 is None:
        raise RuntimeError("Model outputs do not contain 'preds'.")
    # For node vector tasks in ESCGNNModular: (N, 1, d) -> (N, d)
    if preds0.dim() == 3 and preds0.size(1) == 1:
        preds0_vec = preds0.squeeze(1)
    else:
        preds0_vec = preds0

    # Create a rotated clone of the batch
    batch_rot = batch.clone()
    # Random rotation in SO(3)
    R = random_rotation_matrix_3d(device)
    # Rotate the vector feature key (and 'pos' if different)
    rotate_keys = [vector_key]
    if vector_key != 'pos' and hasattr(batch_rot, 'pos'):
        rotate_keys.append('pos')
    apply_rotation_to_batch(batch_rot, R, rotate_keys)

    # Recompute P/Q on rotated sample using dataset config if available
    if cfg is not None and hasattr(cfg, 'dataset_config'):
        ds = cfg.dataset_config
        batch_rot = process_pyg_data(
            batch_rot,
            vector_feat_key=vector_key,
            device=getattr(batch_rot, vector_key).device,
            return_data_object=True,
            graph_construction=None,
            use_mean_recentering=getattr(ds, 'use_mean_recentering', False),
            sing_vect_align_method=getattr(ds, 'sing_vect_align_method', 'column_dot'),
            local_pca_kernel_fn_kwargs={
                'kernel': getattr(ds, 'local_pca_distance_kernel', 'gaussian'),
                'gaussian_eps': getattr(ds, 'local_pca_distance_kernel_scale', None),
            },
            hdf5_tensor_dtype=getattr(ds, 'hdf5_tensor_dtype', 'float16'),
        )

    # Prediction on rotated input (with rotated P/Q)
    outR = model(batch_rot)
    predsR = outR.get('preds', None)
    if predsR is None:
        raise RuntimeError("Model outputs do not contain 'preds' for rotated batch.")
    if predsR.dim() == 3 and predsR.size(1) == 1:
        predsR_vec = predsR.squeeze(1)
    else:
        predsR_vec = predsR

    # Rotate original prediction
    preds0_rot = preds0_vec @ R.T

    # Residuals (filtered) and cosine error
    diff = predsR_vec - preds0_rot
    num = torch.linalg.norm(diff, dim=-1)
    den = torch.linalg.norm(preds0_rot, dim=-1)
    med = torch.median(den)
    eps_abs = 1e-8
    eps_rel = 1e-3 * med
    thresh = torch.maximum(torch.full_like(den, eps_abs), eps_rel)
    mask = den >= thresh
    # Primary (masked) relative error; fallback to epsilon-based mean over all points to avoid NaN
    if mask.any():
        rel = (num[mask] / torch.clamp(den[mask], min=1e-12)).mean().item()
    else:
        rel = (num / torch.clamp(den, min=1e-12)).mean().item()

    dot = (predsR_vec * preds0_rot).sum(dim=-1)
    cos = dot / (torch.clamp(den, min=1e-12) * torch.clamp(torch.linalg.norm(predsR_vec, dim=-1), min=1e-12))
    if mask.any():
        cos_err = (1.0 - torch.clamp(cos[mask], -1.0, 1.0)).mean().item()
    else:
        cos_err = (1.0 - torch.clamp(cos, -1.0, 1.0)).mean().item()

    return {
        'residual_mean_filtered': rel,
        'cosine_error_mean_filtered': cos_err,
        'coverage_fraction': mask.float().mean().item(),
        'residual_l2_mean': num.mean().item(),
        'residual_l2_max': num.max().item(),
    }


@torch.no_grad()
def _get_W_vector_raw(model, batch, device: torch.device):
    if not hasattr(model, 'vector_track_kwargs') or not hasattr(model, '_scatter'):
        return None
    vk = model.vector_track_kwargs.get('feature_key', None)
    ok = model.vector_track_kwargs.get('diffusion_op_key', None)
    if vk is None or ok is None:
        return None
    if not (hasattr(batch, vk) and hasattr(batch, ok)):
        return None
    x = getattr(batch, vk).to(device)
    Q = getattr(batch, ok).to(device)
    Wjxs = model._scatter(
        track='vector',
        x0=x,
        P_or_Q=Q,
        kwargs=model.vector_track_kwargs,
    ) # (N, 1, d, W)
    return Wjxs


@torch.no_grad()
def _apply_vector_mixer_if_any(model, W_vec_raw):
    if W_vec_raw is None:
        return None
    Wv = W_vec_raw
    if getattr(model, 'vector_mixer', None) is None:
        return Wv.permute(0, 3, 1, 2)  # (N, 1, d, W)->(N, W, 1, d)
    WvNCHW = Wv.permute(0, 2, 1, 3)  # (N, d, 1, W)
    WvMixed = model.vector_mixer(WvNCHW)
    return WvMixed.permute(0, 3, 2, 1)  # (N, W', 1, d)


@torch.no_grad()
def _build_frameless_Q_from_P(P_sparse: torch.Tensor, vector_dim: int, device: torch.device) -> torch.Tensor:
    """
    Construct frameless vector operator Q = P ⊗ I_d from scalar P.
    P: (N, N) sparse COO, returns Q: (N*d, N*d) sparse COO.
    """
    P = P_sparse.coalesce()
    idx = P.indices()  # (2, nnz)
    vals = P.values()  # (nnz,)
    N = P.size(0)
    d = int(vector_dim)
    # Repeat indices/values for each coordinate
    i_row = idx[0].repeat_interleave(d)
    i_col = idx[1].repeat_interleave(d)
    coord = torch.arange(d, device=i_row.device)
    coord = coord.repeat(P._nnz())
    row_big = i_row * d + coord
    col_big = i_col * d + coord
    indices_big = torch.stack([row_big, col_big], dim=0)
    values_big = vals.repeat_interleave(d)
    Q = torch.sparse_coo_tensor(indices_big, values_big, (N * d, N * d), device=device)
    return Q.coalesce()


@torch.no_grad()
def check_Q_operator_equivariance(
    batch,
    device: torch.device,
    cfg,
    vector_key: str,
    num_trials: int = 3,
) -> dict:
    """
    Test Q_R (R·v) ≈ R·(Q v) using random vectors v, where
      Q = Q(P, O) from original batch, and Q_R from rotated batch.
    Reports mean/max relative errors over trials.
    """
    batch0 = batch.to(device)
    Rm = random_rotation_matrix_3d(device)
    batchR = batch0.clone()
    rotate_keys = [vector_key]
    if vector_key != 'pos' and hasattr(batchR, 'pos'):
        rotate_keys.append('pos')
    apply_rotation_to_batch(batchR, Rm, rotate_keys)

    ds = getattr(cfg, 'dataset_config', None)
    kernel = get_local_pca_kernel_weights
    kernel_kwargs = {
        'kernel': getattr(ds, 'local_pca_distance_kernel', 'gaussian') if ds is not None else 'gaussian',
        'gaussian_eps': getattr(ds, 'local_pca_distance_kernel_scale', None) if ds is not None else None,
    }
    use_mean_recentering = getattr(ds, 'use_mean_recentering', False) if ds is not None else False
    rank_def_strat = getattr(ds, 'rank_deficiency_strategy', None) if ds is not None else None
    tikh_eps = getattr(ds, 'tikhonov_eps', 1e-3) if ds is not None else 1e-3

    # Build O and Q for original and rotated
    C0 = get_C_i_dict(batch0, vector_key, use_mean_recentering, kernel, kernel_kwargs)
    CR = get_C_i_dict(batchR, vector_key, use_mean_recentering, kernel, kernel_kwargs)
    d = getattr(batch0, vector_key).shape[1]
    O0 = {i: get_l_singular_vecs(C, d, rank_deficiency_strategy=rank_def_strat, tikhonov_eps=tikh_eps) for i, C in C0.items()}
    OR = {i: get_l_singular_vecs(C, d, rank_deficiency_strategy=rank_def_strat, tikhonov_eps=tikh_eps) for i, C in CR.items()}
    P0 = getattr(batch0, 'P').to(device)
    PR = getattr(batchR, 'P').to(device)
    align_method = getattr(ds, 'sing_vect_align_method', 'column_dot') if ds is not None else 'column_dot'
    Q0 = _build_Q_from_P_and_O(P0, O0, mapping='iOjT', align_method=align_method, device=device)
    QR = _build_Q_from_P_and_O(PR, OR, mapping='iOjT', align_method=align_method, device=device)

    # Helpers
    N = getattr(batch0, vector_key).shape[0]
    def apply_block_R(vec_flat: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        v = vec_flat.reshape(N, d)
        vR = v @ R.T
        return vR.reshape(N * d)

    rel_errs = []
    for _ in range(max(1, int(num_trials))):
        v0 = torch.randn(N * d, device=device)
        vR = apply_block_R(v0, Rm)
        outR = torch.sparse.mm(QR, vR.unsqueeze(1)).squeeze(1)
        out0 = torch.sparse.mm(Q0, v0.unsqueeze(1)).squeeze(1)
        out0_rot = apply_block_R(out0, Rm)
        diff = outR - out0_rot
        den = torch.linalg.norm(out0_rot)
        num = torch.linalg.norm(diff)
        rel = (num / torch.clamp(den, min=1e-12)).item()
        rel_errs.append(rel)
    return {
        'Q_equiv_rel_mean': float(sum(rel_errs) / len(rel_errs)),
        'Q_equiv_rel_max': float(max(rel_errs)),
        'trials': len(rel_errs),
    }

@torch.no_grad()
def _get_W_vector_with_Q(model, batch, device: torch.device, Q_override: torch.Tensor):
    """
    Compute vector scattering using a provided Q operator (e.g., frameless P ⊗ I_d).
    Returns shape (N, 1, d, W) like _get_W_vector_raw.
    """
    if not hasattr(model, 'vector_track_kwargs') or not hasattr(model, '_scatter'):
        return None
    vk = model.vector_track_kwargs.get('feature_key', None)
    if vk is None:
        return None
    if not hasattr(batch, vk):
        return None
    x = getattr(batch, vk).to(device)
    Wjxs = model._scatter(
        track='vector',
        x0=x,
        P_or_Q=Q_override,
        kwargs=model.vector_track_kwargs,
    )
    return Wjxs


@torch.no_grad()
def _build_Q_from_P_and_O(
    P_sparse: torch.Tensor,
    O_dict: dict,
    mapping: str,
    align_method: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Build Q from scalar P and local frames O_i using a mapping convention:
      - 'iOjT': block(i,j) = p_ij * (O_i @ O_j^T)
      - 'jOiT': block(i,j) = p_ij * (O_j @ O_i^T)
    """
    P = P_sparse.coalesce()
    idx = P.indices()  # (2, nnz)
    vals = P.values()  # (nnz,)
    nnz = P._nnz()
    # Infer d from first O in dict
    any_key = next(iter(O_dict))
    d = O_dict[any_key].shape[0]
    N = P.size(0)

    # Prepare indices and values lists
    blocks_vals = []
    blocks_idx = []
    # Base block indices for a dxd block at origin
    base_r = torch.repeat_interleave(torch.arange(d, device=device), d)
    base_c = torch.tile(torch.arange(d, device=device), (d,))
    for k in range(nnz):
        i = int(idx[0, k].item())
        j = int(idx[1, k].item())
        p_ij = vals[k]
        if (i not in O_dict) or (j not in O_dict):
            continue
        if mapping == 'iOjT':
            Bij = calc_O_ij(O_dict[i], O_dict[j], enforce_sign=True, sing_vect_align_method=align_method)
        else:
            Bij = calc_O_ij(O_dict[j], O_dict[i], enforce_sign=True, sing_vect_align_method=align_method)
        Bij = (p_ij * Bij).reshape(-1)
        blocks_vals.append(Bij)
        row_big = base_r + d * i
        col_big = base_c + d * j
        blocks_idx.append(torch.stack([row_big, col_big], dim=0))
    if len(blocks_vals) == 0:
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, device=device),
            (N * d, N * d),
            device=device,
        )
    values_big = torch.cat(blocks_vals)
    indices_big = torch.cat(blocks_idx, dim=1)
    Q = torch.sparse_coo_tensor(indices_big, values_big, (N * d, N * d), device=device)
    return Q.coalesce()

@torch.no_grad()
def check_Oi_equivariance(
    batch,
    device: torch.device,
    cfg,
    vector_key: str,
    use_mean_recentering: bool = False,
) -> dict:
    """
    Check per-node local frame equivariance: O_i(R·x) ≈ R·O_i(x) up to column sign flips.
    Returns mean/max Frobenius errors after column-wise dot-product sign alignment.
    """
    batch0 = batch.to(device)
    # Build rotated clone and recompute graph-dependent structures (P/Q untouched here)
    Rm = random_rotation_matrix_3d(device)
    batchR = batch0.clone()
    rotate_keys = [vector_key]
    if vector_key != 'pos' and hasattr(batchR, 'pos'):
        rotate_keys.append('pos')
    apply_rotation_to_batch(batchR, Rm, rotate_keys)

    # Compute C_i and O_i on both batches
    ds = getattr(cfg, 'dataset_config', None)
    kernel = get_local_pca_kernel_weights
    kernel_kwargs = {
        'kernel': getattr(ds, 'local_pca_distance_kernel', 'gaussian') if ds is not None else 'gaussian',
        'gaussian_eps': getattr(ds, 'local_pca_distance_kernel_scale', None) if ds is not None else None,
    }
    use_mean_recentering = getattr(ds, 'use_mean_recentering', use_mean_recentering) if ds is not None else use_mean_recentering
    rank_def_strat = getattr(ds, 'rank_deficiency_strategy', None) if ds is not None else None
    tikh_eps = getattr(ds, 'tikhonov_eps', 1e-3) if ds is not None else 1e-3

    C0 = get_C_i_dict(batch0, vector_key, use_mean_recentering, kernel, kernel_kwargs)
    CR = get_C_i_dict(batchR, vector_key, use_mean_recentering, kernel, kernel_kwargs)
    d = getattr(batch0, vector_key).shape[1]
    O0 = {i: get_l_singular_vecs(C, d, rank_deficiency_strategy=rank_def_strat, tikhonov_eps=tikh_eps) for i, C in C0.items()}
    OR = {i: get_l_singular_vecs(C, d, rank_deficiency_strategy=rank_def_strat, tikhonov_eps=tikh_eps) for i, C in CR.items()}

    # Compare OR[i] vs Rm @ O0[i] with column-wise sign alignment
    errs = []
    for i in O0.keys():
        Oexp = Rm @ O0[i]
        Oact = OR[i]
        Oalign = match_B_col_directions_to_A(Oexp, Oact)
        diff = Oalign - Oexp
        err = torch.linalg.norm(diff, ord='fro') / max(d, 1)
        errs.append(err.item())
    import math as _math
    mean_err = float(sum(errs) / max(len(errs), 1))
    max_err = float(max(errs) if errs else 0.0)
    return {
        'Oi_dot_align_frob_mean': mean_err,
        'Oi_dot_align_frob_max': max_err,
    }


@torch.no_grad()
def check_Oij_equivariance(
    batch,
    device: torch.device,
    cfg,
    vector_key: str,
) -> dict:
    """
    Check pairwise frame transport equivariance: O_ij(R·x) ≈ R·O_ij(x)·R^T.
    O_ij is built with column-dot alignment (same as training pipeline).
    Reports mean/max Frobenius error over edges.
    """
    batch0 = batch.to(device)
    Rm = random_rotation_matrix_3d(device)
    batchR = batch0.clone()
    rotate_keys = [vector_key]
    if vector_key != 'pos' and hasattr(batchR, 'pos'):
        rotate_keys.append('pos')
    apply_rotation_to_batch(batchR, Rm, rotate_keys)

    ds = getattr(cfg, 'dataset_config', None)
    kernel = get_local_pca_kernel_weights
    kernel_kwargs = {
        'kernel': getattr(ds, 'local_pca_distance_kernel', 'gaussian') if ds is not None else 'gaussian',
        'gaussian_eps': getattr(ds, 'local_pca_distance_kernel_scale', None) if ds is not None else None,
    }
    use_mean_recentering = getattr(ds, 'use_mean_recentering', False) if ds is not None else False
    rank_def_strat = getattr(ds, 'rank_deficiency_strategy', None) if ds is not None else None
    tikh_eps = getattr(ds, 'tikhonov_eps', 1e-3) if ds is not None else 1e-3
    align_method = getattr(ds, 'sing_vect_align_method', 'column_dot') if ds is not None else 'column_dot'

    C0 = get_C_i_dict(batch0, vector_key, use_mean_recentering, kernel, kernel_kwargs)
    CR = get_C_i_dict(batchR, vector_key, use_mean_recentering, kernel, kernel_kwargs)
    d = getattr(batch0, vector_key).shape[1]
    O0 = {i: get_l_singular_vecs(C, d, rank_deficiency_strategy=rank_def_strat, tikhonov_eps=tikh_eps) for i, C in C0.items()}
    OR = {i: get_l_singular_vecs(C, d, rank_deficiency_strategy=rank_def_strat, tikhonov_eps=tikh_eps) for i, C in CR.items()}

    # Iterate over edges
    if not hasattr(batch0, 'edge_index'):
        return {'Oij_frob_mean': float('nan'), 'Oij_frob_max': float('nan')}
    edge_index = batch0.edge_index
    errs = []
    for k in range(edge_index.shape[1]):
        i = int(edge_index[0, k].item())
        j = int(edge_index[1, k].item())
        if (i not in O0) or (j not in O0) or (i not in OR) or (j not in OR):
            continue
        Oij0 = calc_O_ij(O0[i], O0[j], enforce_sign=True, sing_vect_align_method=align_method)
        OijR = calc_O_ij(OR[i], OR[j], enforce_sign=True, sing_vect_align_method=align_method)
        target = Rm @ Oij0 @ Rm.T
        diff = OijR - target
        err = torch.linalg.norm(diff, ord='fro') / max(d, 1)
        errs.append(err.item())
    mean_err = float(sum(errs) / max(len(errs), 1))
    max_err = float(max(errs) if errs else 0.0)
    return {
        'Oij_frob_mean': mean_err,
        'Oij_frob_max': max_err,
        'num_pairs': len(errs),
    }


@torch.no_grad()
def _residual_for_stack_of_vectors(stack_rot, stack_base, R: torch.Tensor) -> dict:
    N, W, d = stack_base.shape
    base = stack_base.reshape(N * W, d)
    base_rot_expected = base @ R.T
    base_rot_expected = base_rot_expected.reshape(N, W, d)
    diff = stack_rot - base_rot_expected
    num = torch.linalg.norm(diff, dim=-1)          # (N, W)
    den = torch.linalg.norm(base_rot_expected, dim=-1)  # (N, W)
    med = torch.median(den)
    eps_abs = 1e-8
    eps_rel = 1e-3 * med
    thresh = torch.maximum(torch.full_like(den, eps_abs), eps_rel)
    mask = den >= thresh

    # Cosine error
    dot = (stack_rot * base_rot_expected).sum(dim=-1)
    cos = dot / (torch.clamp(den, min=1e-12) * torch.clamp(torch.linalg.norm(stack_rot, dim=-1), min=1e-12))
    cos = torch.clamp(cos, -1.0, 1.0)

    # Per-wavelet means over nodes
    rel_list, cos_list, cov_list = [], [], []
    for k in range(W):
        m = mask[:, k]
        cov_list.append(m.float().mean().item())
        if m.any():
            rel_list.append((num[m, k] / torch.clamp(den[m, k], min=1e-12)).mean().item())
            cos_list.append((1.0 - cos[m, k]).mean().item())
    # Fallbacks to avoid NaNs when coverage is zero across all wavelets
    rel_mean = float('nan') if len(rel_list) == 0 else sum(rel_list) / len(rel_list)
    cos_mean = float('nan') if len(cos_list) == 0 else sum(cos_list) / len(cos_list)
    if not torch.isfinite(torch.tensor(rel_mean)):
        rel_mean = (num / torch.clamp(den, min=1e-12)).mean().item()
    if not torch.isfinite(torch.tensor(cos_mean)):
        cos = torch.clamp(cos, -1.0, 1.0)
        cos_mean = (1.0 - cos).mean().item()
    coverage = 0.0 if len(cov_list) == 0 else sum(cov_list) / len(cov_list)
    return {
        'rel_mean_per_wavelet': rel_mean,
        'cos_err_mean_per_wavelet': cos_mean,
        'coverage_fraction': coverage,
    }


@torch.no_grad()
def stagewise_equivariance(model, batch, device: torch.device, cfg=None) -> dict:
    if hasattr(model, 'run_epoch_zero_methods'):
        model.run_epoch_zero_methods(batch.to(device))

    R = random_rotation_matrix_3d(device)

    batch0 = batch.to(device)
    batchR = batch0.clone()
    vk = getattr(model, 'vector_track_kwargs', {}).get('feature_key', 'pos')
    rotate_keys = [vk]
    if vk != 'pos' and hasattr(batchR, 'pos'):
        rotate_keys.append('pos')
    apply_rotation_to_batch(batchR, R, rotate_keys)
    # Recompute P/Q on rotated sample
    if cfg is not None and hasattr(cfg, 'dataset_config'):
        ds = cfg.dataset_config
        batchR = process_pyg_data(
            batchR,
            vector_feat_key=rotate_keys[0],
            device=getattr(batchR, rotate_keys[0]).device,
            return_data_object=True,
            graph_construction=None,
            use_mean_recentering=getattr(ds, 'use_mean_recentering', False),
            sing_vect_align_method=getattr(ds, 'sing_vect_align_method', 'column_dot'),
            local_pca_kernel_fn_kwargs={
                'kernel': getattr(ds, 'local_pca_distance_kernel', 'gaussian'),
                'gaussian_eps': getattr(ds, 'local_pca_distance_kernel_scale', None),
            },
            hdf5_tensor_dtype=getattr(ds, 'hdf5_tensor_dtype', 'float16'),
        )

    # Raw W
    W0 = _get_W_vector_raw(model, batch0, device)
    WR = _get_W_vector_raw(model, batchR, device)
    res_raw = None
    res_raw_by_order = None
    if (W0 is not None) and (WR is not None):
        S0 = W0.squeeze(1).permute(0, 2, 1)  # (N, W, d)
        SR = WR.squeeze(1).permute(0, 2, 1)
        res_raw = _residual_for_stack_of_vectors(SR, S0, R)

        # Break down by scattering order using W layout from ESCGNNModular._scatter
        # Vector track layout: [W0 (1), W1 (n), W2 (n*(n-1)/2) if enabled]
        Wdim = S0.shape[1]
        has_second = not getattr(model, 'ablate_second_order_wavelets', False)
        n1 = None
        if has_second:
            # Solve n1*(n1+1)/2 = Wdim - 1
            rhs = max(int(Wdim - 1), 0)
            # Quadratic solution: n = floor((-1 + sqrt(1 + 8*rhs)) / 2)
            n_candidate = int(((-1.0 + (1.0 + 8.0 * rhs) ** 0.5) // 1))
            # Adjust if off-by-one
            found = False
            for k in [n_candidate - 1, n_candidate, n_candidate + 1, n_candidate + 2]:
                if k >= 0 and (k * (k + 1)) // 2 == rhs:
                    n1 = k
                    found = True
                    break
            if not found:
                # Fallback: brute force small k
                for k in range(0, Wdim + 1):
                    if (k * (k + 1)) // 2 == rhs:
                        n1 = k
                        found = True
                        break
            if not found:
                n1 = max(Wdim - 1, 0)
        else:
            n1 = max(Wdim - 1, 0)

        idx0_end = 1
        idx1_end = 1 + n1
        # Order 0
        res0 = _residual_for_stack_of_vectors(SR[:, 0:idx0_end, :], S0[:, 0:idx0_end, :], R)
        # Order 1
        res1 = None
        if idx1_end > idx0_end:
            res1 = _residual_for_stack_of_vectors(SR[:, idx0_end:idx1_end, :], S0[:, idx0_end:idx1_end, :], R)
        # Order 2
        res2 = None
        if has_second and idx1_end < Wdim:
            res2 = _residual_for_stack_of_vectors(SR[:, idx1_end:, :], S0[:, idx1_end:, :], R)

        res_raw_by_order = {
            'order0': res0,
            'order1': res1,
            'order2': res2,
        }

    # Frameless Q = P ⊗ I_d diagnostic
    res_raw_frameless = None
    res_raw_frameless_by_order = None
    res_mix_frameless = None
    res_sum_frameless = None
    if hasattr(batch0, 'P') and hasattr(batchR, 'P') \
    and (getattr(model, 'vector_track_kwargs', None) is not None):
        d_vec = int(model.vector_track_kwargs.get('vector_dim', 3))
        P0 = getattr(batch0, 'P').to(device)
        PR = getattr(batchR, 'P').to(device)
        Q0_fr = _build_frameless_Q_from_P(P0, d_vec, device)
        QR_fr = _build_frameless_Q_from_P(PR, d_vec, device)
        W0_fr = _get_W_vector_with_Q(model, batch0, device, Q0_fr)
        WR_fr = _get_W_vector_with_Q(model, batchR, device, QR_fr)
        if (W0_fr is not None) and (WR_fr is not None):
            S0f = W0_fr.squeeze(1).permute(0, 2, 1)
            SRf = WR_fr.squeeze(1).permute(0, 2, 1)
            res_raw_frameless = _residual_for_stack_of_vectors(SRf, S0f, R)
            # By order
            Wdim = S0f.shape[1]
            has_second = not getattr(model, 'ablate_second_order_wavelets', False)
            n1 = None
            if has_second:
                rhs = max(int(Wdim - 1), 0)
                n_candidate = int(((-1.0 + (1.0 + 8.0 * rhs) ** 0.5) // 1))
                found = False
                for k in [n_candidate - 1, n_candidate, n_candidate + 1, n_candidate + 2]:
                    if k >= 0 and (k * (k + 1)) // 2 == rhs:
                        n1 = k
                        found = True
                        break
                if not found:
                    for k in range(0, Wdim + 1):
                        if (k * (k + 1)) // 2 == rhs:
                            n1 = k
                            found = True
                            break
                if not found:
                    n1 = max(Wdim - 1, 0)
            else:
                n1 = max(Wdim - 1, 0)
            idx0_end = 1
            idx1_end = 1 + n1
            res0f = _residual_for_stack_of_vectors(SRf[:, 0:idx0_end, :], S0f[:, 0:idx0_end, :], R)
            res1f = None
            if idx1_end > idx0_end:
                res1f = _residual_for_stack_of_vectors(SRf[:, idx0_end:idx1_end, :], S0f[:, idx0_end:idx1_end, :], R)
            res2f = None
            if has_second and idx1_end < Wdim:
                res2f = _residual_for_stack_of_vectors(SRf[:, idx1_end:, :], S0f[:, idx1_end:, :], R)
            res_raw_frameless_by_order = {
                'order0': res0f,
                'order1': res1f,
                'order2': res2f,
            }
            # Mixed frameless (use same mixer)
            W0m_fr = _apply_vector_mixer_if_any(model, W0_fr)
            WRm_fr = _apply_vector_mixer_if_any(model, WR_fr)
            if (W0m_fr is not None) and (WRm_fr is not None):
                S0mf = W0m_fr.squeeze(2)
                SRmf = WRm_fr.squeeze(2)
                res_mix_frameless = _residual_for_stack_of_vectors(SRmf, S0mf, R)
                # Uniform sum frameless
                Nf, Wpf, _, df = W0m_fr.shape
                gatesf = torch.full((Nf, Wpf), 1.0 / max(Wpf, 1), device=device)
                v0f = (W0m_fr.squeeze(2) * gatesf.unsqueeze(-1)).sum(dim=1)
                vRf = (WRm_fr.squeeze(2) * gatesf.unsqueeze(-1)).sum(dim=1)
                v0f_rot = v0f @ R.T
                difff = vRf - v0f_rot
                numf = torch.linalg.norm(difff, dim=-1)
                denf = torch.linalg.norm(v0f_rot, dim=-1)
                medf = torch.median(denf)
                eps_abs = 1e-8
                eps_rel = 1e-3 * medf
                threshf = torch.maximum(torch.full_like(denf, eps_abs), eps_rel)
                maskf = denf >= threshf
                relf = (numf[maskf] / torch.clamp(denf[maskf], min=1e-12)).mean().item() if maskf.any() else (numf / torch.clamp(denf, min=1e-12)).mean().item()
                dotf = (vRf * v0f_rot).sum(dim=-1)
                cosf = dotf / (torch.clamp(denf, min=1e-12) * torch.clamp(torch.linalg.norm(vRf, dim=-1), min=1e-12))
                cos_errf = (1.0 - torch.clamp(cosf[maskf], -1.0, 1.0)).mean().item() if maskf.any() else (1.0 - torch.clamp(cosf, -1.0, 1.0)).mean().item()
                res_sum_frameless = {
                    'rel_mean_filtered': relf,
                    'cos_err_mean_filtered': cos_errf,
                    'coverage_fraction': maskf.float().mean().item(),
                }
    # After mixer
    W0m = _apply_vector_mixer_if_any(model, W0)
    WRm = _apply_vector_mixer_if_any(model, WR)
    res_mix = None
    if (W0m is not None) and (WRm is not None):
        S0m = W0m.squeeze(2)  # (N, W', d)
        SRm = WRm.squeeze(2)
        res_mix = _residual_for_stack_of_vectors(SRm, S0m, R)

    # Uniform gated sum
    res_sum = None
    if (W0m is not None) and (WRm is not None):
        N, Wp, _, d = W0m.shape
        gates = torch.full((N, Wp), 1.0 / max(Wp, 1), device=device)
        v0 = (W0m.squeeze(2) * gates.unsqueeze(-1)).sum(dim=1)
        vR = (WRm.squeeze(2) * gates.unsqueeze(-1)).sum(dim=1)
        v0_rot = v0 @ R.T
        diff = vR - v0_rot
        num = torch.linalg.norm(diff, dim=-1)
        den = torch.linalg.norm(v0_rot, dim=-1)
        med = torch.median(den)
        eps_abs = 1e-8
        eps_rel = 1e-3 * med
        thresh = torch.maximum(torch.full_like(den, eps_abs), eps_rel)
        mask = den >= thresh
        rel = (num[mask] / torch.clamp(den[mask], min=1e-12)).mean().item() if mask.any() else float('nan')
        dot = (vR * v0_rot).sum(dim=-1)
        cos = dot / (torch.clamp(den, min=1e-12) * torch.clamp(torch.linalg.norm(vR, dim=-1), min=1e-12))
        cos_err = (1.0 - torch.clamp(cos[mask], -1.0, 1.0)).mean().item() if mask.any() else float('nan')
        # Fallbacks when coverage is zero
        if not mask.any():
            rel = (num / torch.clamp(den, min=1e-12)).mean().item()
            cos = torch.clamp(cos, -1.0, 1.0)
            cos_err = (1.0 - cos).mean().item()
        res_sum = {
            'rel_mean_filtered': rel,
            'cos_err_mean_filtered': cos_err,
            'coverage_fraction': mask.float().mean().item(),
        }

    return {
        'raw_W': res_raw,
        'raw_W_by_order': res_raw_by_order,
        'raw_W_frameless': res_raw_frameless,
        'raw_W_frameless_by_order': res_raw_frameless_by_order,
        'mixed_W': res_mix,
        'mixed_W_frameless': res_mix_frameless,
        'uniform_sum': res_sum,
        'uniform_sum_frameless': res_sum_frameless,
    }


def main():
    clargs = get_clargs()
    config_manager = ConfigManager(clargs)
    config = config_manager.config

    # Only meaningful for node-level vector tasks
    if 'node' not in config.dataset_config.task or 'vector' not in config.dataset_config.task:
        print(f"Task '{config.dataset_config.task}' is not node-level vector; exiting.")
        return

    acc = Accelerator(device_placement=False if config.using_pytorch_geo else config.device,
                      mixed_precision='no' if config.mixed_precision == 'none' else config.mixed_precision)

    dataset = load_dataset(config, model_key=config.model_config.model_key)
    if dataset is None:
        raise RuntimeError("Dataset failed to load.")

    # Simple random split
    from data_processing.data_utilities import get_random_splits
    splits = get_random_splits(n=len(dataset), seed=config.dataset_config.split_seed,
                               train_prop=config.dataset_config.train_prop,
                               valid_prop=config.dataset_config.valid_prop)

    dloaders, config = create_dataloaders(dataset, splits, config)

    # Prepare model
    config, model, dloaders = prepare_escgnn_model(config, dloaders, acc=acc)
    model.to(acc.device)
    model.eval()

    # Get one batch from valid if available, else train
    dl = dloaders.get('valid', None) or dloaders['train']
    batch = next(iter(dl))

    # Compute residual
    vec_key = config.dataset_config.vector_feat_key
    res = compute_equivariance_residual(model, batch, vec_key, acc.device, cfg=config)
    print("Equivariance residuals:")
    for k, v in res.items():
        print(f"  {k}: {v:.4e}")

    print("\nStage-wise residuals (vector track):")
    stages = stagewise_equivariance(model, batch, acc.device, cfg=config)
    for stage, metrics in stages.items():
        if metrics is None:
            print(f"  {stage}: None")
            continue
        print(f"  {stage}:")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4e}")
                elif isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.item():.4e}")
                else:
                    # Pretty-print nested dicts like order splits with .4e formatting for floats
                    if isinstance(v, dict):
                        # Reformat floats inside nested dict
                        def _fmt(obj):
                            if isinstance(obj, float):
                                return f"{obj:.4e}"
                            return obj
                        pretty = {kk: _fmt(vv) for kk, vv in v.items()}
                        print(f"    {k}: {pretty}")
                    else:
                        print(f"    {k}: {v}")
        else:
            print(f"    rel_mean_filtered: {metrics:.4e}")

    # Local O_i equivariance check
    print(f"\nLocal frame O_i equivariance ({getattr(config.dataset_config, 'sing_vect_align_method', 'column_dot')}-aligned):")
    oi_stats = check_Oi_equivariance(batch, acc.device, config, vec_key)
    for k, v in oi_stats.items():
        print(f"  {k}: {v:.4e}")

    # O_ij equivariance check
    print("\nO_ij transport equivariance (target: R·O_ij·R^T):")
    oij_stats = check_Oij_equivariance(batch, acc.device, config, vec_key)
    for k, v in oij_stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4e}")
        else:
            print(f"  {k}: {v}")

    # Q block placement sanity: build Q from P and O with both mappings and compare scattering
    print("\nQ construction sanity (iOjT vs jOiT):")
    # Recompute O on the batch (unrotated)
    ds = getattr(config, 'dataset_config', None)
    kernel = get_local_pca_kernel_weights
    kernel_kwargs = {
        'kernel': getattr(ds, 'local_pca_distance_kernel', 'gaussian') if ds is not None else 'gaussian',
        'gaussian_eps': getattr(ds, 'local_pca_distance_kernel_scale', None) if ds is not None else None,
    }
    use_mean_recentering = getattr(ds, 'use_mean_recentering', False) if ds is not None else False
    rank_def_strat = getattr(ds, 'rank_deficiency_strategy', None) if ds is not None else None
    tikh_eps = getattr(ds, 'tikhonov_eps', 1e-3) if ds is not None else 1e-3
    C0 = get_C_i_dict(batch, vec_key, use_mean_recentering, kernel, kernel_kwargs)
    d = getattr(batch, vec_key).shape[1]
    O0 = {i: get_l_singular_vecs(C, d, rank_deficiency_strategy=rank_def_strat, tikhonov_eps=tikh_eps) for i, C in C0.items()}
    P0 = getattr(batch, 'P').to(acc.device)
    align_method = getattr(config.dataset_config, 'sing_vect_align_method', 'column_dot')
    print(f"singular vector align_method: {align_method}")
    QiOjT = _build_Q_from_P_and_O(P0, O0, mapping='iOjT', align_method=align_method, device=acc.device)
    QjOiT = _build_Q_from_P_and_O(P0, O0, mapping='jOiT', align_method=align_method, device=acc.device)
    # Scatter with both and compare to model’s vector scatter
    W_model = _get_W_vector_raw(model, batch, acc.device)
    W_iOjT = _get_W_vector_with_Q(model, batch, acc.device, QiOjT)
    W_jOiT = _get_W_vector_with_Q(model, batch, acc.device, QjOiT)
    def _stack_err(Wa, Wb):
        if Wa is None or Wb is None:
            return float('nan')
        Sa = Wa.squeeze(1).permute(0, 2, 1)
        Sb = Wb.squeeze(1).permute(0, 2, 1)
        diff = Sa - Sb
        return float(torch.linalg.norm(diff) / max(diff.numel(), 1))
    print(f"  ||W(model Q) - W(P⊗I)||/size: {_stack_err(W_model, _get_W_vector_with_Q(model, batch, acc.device, _build_frameless_Q_from_P(P0, d, acc.device))):.4e}")
    print(f"  ||W(iOjT) - W(P⊗I)||/size: {_stack_err(W_iOjT, _get_W_vector_with_Q(model, batch, acc.device, _build_frameless_Q_from_P(P0, d, acc.device))):.4e}")
    print(f"  ||W(jOiT) - W(P⊗I)||/size: {_stack_err(W_jOiT, _get_W_vector_with_Q(model, batch, acc.device, _build_frameless_Q_from_P(P0, d, acc.device))):.4e}")

    # Direct Q-operator equivariance (random vector tests)
    print("\nQ operator equivariance (random vector tests):")
    qop_stats = check_Q_operator_equivariance(batch, acc.device, config, vec_key)
    for k, v in qop_stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4e}")
        else:
            print(f"  {k}: {v}")

    # Q(model) vs Q(built) sanity
    print("\nQ(model) vs Q(built from P,O):")
    try:
        ds = getattr(config, 'dataset_config', None)
        align_method = getattr(ds, 'sing_vect_align_method', 'column_dot') if ds is not None else 'column_dot'
        kernel = get_local_pca_kernel_weights
        kernel_kwargs = {
            'kernel': getattr(ds, 'local_pca_distance_kernel', 'gaussian') if ds is not None else 'gaussian',
            'gaussian_eps': getattr(ds, 'local_pca_distance_kernel_scale', None) if ds is not None else None,
        }
        use_mean_recentering = getattr(ds, 'use_mean_recentering', False) if ds is not None else False
        rank_def_strat = getattr(ds, 'rank_deficiency_strategy', None) if ds is not None else None
        tikh_eps = getattr(ds, 'tikhonov_eps', 1e-3) if ds is not None else 1e-3

        P0 = getattr(batch, 'P', None)
        Qm = getattr(batch, 'Q', None)
        if (P0 is None) or (Qm is None):
            # Build P/Q on the original batch
            bproc = process_pyg_data(
                batch,
                vector_feat_key=vec_key,
                device=getattr(batch, vec_key).device,
                return_data_object=True,
                graph_construction=None,
                use_mean_recentering=use_mean_recentering,
                sing_vect_align_method=align_method,
                rank_deficiency_strategy=rank_def_strat,
                tikhonov_eps=tikh_eps,
                local_pca_kernel_fn_kwargs=kernel_kwargs,
                hdf5_tensor_dtype=getattr(ds, 'hdf5_tensor_dtype', 'float16') if ds is not None else 'float16',
            )
            P0 = bproc.P
            Qm = bproc.Q
        # Build Q from P and O
        C0 = get_C_i_dict(batch, vec_key, use_mean_recentering, kernel, kernel_kwargs)
        d = getattr(batch, vec_key).shape[1]
        O0 = {i: get_l_singular_vecs(C, d, rank_deficiency_strategy=rank_def_strat, tikhonov_eps=tikh_eps) for i, C in C0.items()}
        Qb = _build_Q_from_P_and_O(P0.to(acc.device), O0, mapping='iOjT', align_method=align_method, device=acc.device)
        # Relative Frobenius difference (dense for simplicity in this diagnostic)
        Qb_dense = Qb.to_dense()
        Qm_dense = Qm.to(acc.device).to_dense()
        num = torch.linalg.norm(Qm_dense - Qb_dense)
        den = torch.linalg.norm(Qb_dense)
        rel = (num / torch.clamp(den, min=1e-12)).item()
        print(f"  rel_fro: {rel:.4e}")
    except Exception as e:
        print(f"  skipped (error: {e})")


if __name__ == '__main__':
    main()


