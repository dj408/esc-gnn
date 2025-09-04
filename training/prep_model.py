import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader as PyGDataLoader
from typing import Dict, Any, Optional, Tuple
from config.model_config import ModelConfig
from config.dataset_config import DatasetConfig
from config.train_config import TrainingConfig
from models.escgnn import ESCGNN
from models.escgnn_modular import ESCGNNModular
from models.escgnn_radial import ESCGNNRadial
from models.vanilla_nn import VanillaNN
from models.comparisons.comparison_module import ComparisonModel
from models.comparisons.egnn import EGNNModel
from models.comparisons.tfn import TFNModel
from models.comparisons.legs import LEGSModel
from models.comparisons.pyg_models import PyGModel
import time
from accelerate import Accelerator

from models.class_maps import (
    SCALAR_NONLIN_FN_MAP,
    MLP_NONLIN_MODULE_MAP,
    LOSS_FN_MAP,
)


# ============================================================
#               SHARED HELPER FUNCTIONS             
# ============================================================
def _process_scales(scales):
    """
    Convert various user-provided formats to a tensor or None.

    Args:
        scales: 'dyadic', list[int], list[list[int]], or torch.Tensor.
    Returns:
        torch.LongTensor or None (for 'dyadic').
    """
    if isinstance(scales, str):
        # The default string ('dyadic') signals to use dyadic scales.
        return None
    if isinstance(scales, torch.Tensor):
        return scales.long()
    if isinstance(scales, list):
        try:
            return torch.tensor(scales, dtype=torch.long)
        except Exception:
            # Fallback: let torch infer dtype (handles nested list lengths)
            return torch.tensor(scales)
    # Unsupported type – fall back to dyadic behavior
    return None


def _setup_base_module_kwargs(
    device: torch.device,
    config: TrainingConfig, 
    dataset_config: DatasetConfig
) -> dict:
    """
    Set up base module keyword arguments for ESCGNN models.
    
    Args:
        config: TrainingConfig object
        dataset_config: DatasetConfig object
    Returns:
        Dictionary of base module keyword arguments
    """
    base_module_kwargs = {
        'device': device,
        'task': dataset_config.task,
        'metrics_kwargs': {
            'num_outputs': dataset_config.target_dim
        },
        'verbosity': config.verbosity
    }
    
    # Map loss function key to actual PyTorch loss function
    if hasattr(config, 'loss_fn') and config.loss_fn in LOSS_FN_MAP:
        base_module_kwargs['loss_fn'] = LOSS_FN_MAP[config.loss_fn]
        # Set loss function kwargs with mean reduction for all loss functions
        if config.loss_fn == 'huber':
            base_module_kwargs['loss_fn_kwargs'] = {
                'beta': getattr(config, 'huber_delta'),
                'reduction': 'mean',
            }
        else:
            # For MSE and L1 loss, ensure mean reduction
            base_module_kwargs['loss_fn_kwargs'] = {
                'reduction': 'mean',
            }
    
    # If specified, pass target normalization statistics to base module
    # (for single vs. multi-target data, the dim of these params is
    # processed upstream, in prep_dataset.py)
    if hasattr(dataset_config, 'target_preprocessing_type') \
    and dataset_config.target_preprocessing_type is not None:
        if hasattr(config.dataset_config, 'target_preproc_stats') \
        and config.dataset_config.target_preproc_stats is not None:
            target_preproc_stats = config.dataset_config.target_preproc_stats
            stats = {
                'center': torch.tensor(target_preproc_stats['center']), 
                'scale': torch.tensor(target_preproc_stats['scale'])
            }
            base_module_kwargs['has_normalized_train_targets'] = True
            base_module_kwargs['target_preproc_stats'] = stats
        else:
            raise ValueError(
                f"Target preprocessing type specified as '{dataset_config.target_preprocessing_type}', "
                f"but no target preprocessing stats found in config!"
            )
    else:
        base_module_kwargs['has_normalized_train_targets'] = False
        base_module_kwargs['target_preproc_stats'] = None
        
    return base_module_kwargs


def _setup_mlp_kwargs(
    model_config: ModelConfig, 
    dataset_config: DatasetConfig
) -> dict:
    """
    Set up MLP keyword arguments for ESCGNN models.
    
    Args:
        model_config: ModelConfig object
        dataset_config: DatasetConfig object
        
    Returns:
        Dictionary of MLP keyword arguments
    """
    mlp_kwargs = {
        'hidden_dims_list': model_config.mlp_hidden_dim,
        'output_dim': dataset_config.target_dim,
        'use_batch_normalization': model_config.mlp_use_batch_normalization,
    }
    
    # Propagate dropout/batch norm parameters (for VanillaNN)
    if model_config.mlp_dropout_p is not None:
        mlp_kwargs['use_dropout'] = True
        mlp_kwargs['dropout_p'] = model_config.mlp_dropout_p
        mlp_kwargs['use_batch_normalization'] = model_config.mlp_use_batch_normalization
    else:
        mlp_kwargs['use_dropout'] = False
    
    return mlp_kwargs


def _setup_ablation_flags(
    config: TrainingConfig, 
    dataset_config: DatasetConfig
) -> tuple:
    """
    Set up ablation flags and propagate them to dataset config.
    
    Args:
        config: TrainingConfig object
        dataset_config: DatasetConfig object
    Returns:
        Tuple of (ablate_vector_track, ablate_scalar_track)
    """
    ablate_vector_track = getattr(config, 'ablate_vector_track')
    ablate_scalar_track = getattr(config, 'ablate_scalar_track')
    # Propagate dynamically so data prepping sees it (even though not a dataclass field)
    setattr(dataset_config, 'ablate_vector_track', ablate_vector_track)
    setattr(dataset_config, 'ablate_scalar_track', ablate_scalar_track)
    
    return ablate_vector_track, ablate_scalar_track


def _print_model_settings(
    config: TrainingConfig, 
    scalar_custom: bool, 
    vector_custom: bool, 
    scalar_scales: Optional[torch.Tensor] = None, 
    vector_scales: Optional[torch.Tensor] = None, 
    model_name: str = "ESCGNN", 
    acc: Optional[Accelerator] = None
) -> None:
    """
    Print model settings confirmations (only once on main process).
    
    Args:
        config: TrainingConfig object
        scalar_custom: Whether scalar scales are custom
        vector_custom: Whether vector scales are custom
        scalar_scales: Scalar scales tensor/list
        vector_scales: Vector scales tensor/list
        model_name: Name of the model for logging
        acc: Optional Accelerator object
    """
    if (acc is None) or (acc.is_main_process):
        # Print confirmation of custom diffusion scales (only once on main process)
        if scalar_custom or vector_custom:
            msg_parts = []
            if scalar_custom:
                _scales = scalar_scales
                num_scales = _scales.numel() if isinstance(_scales, torch.Tensor) else len(_scales)
                msg_parts.append(f"scalar (n_scales={num_scales})")
            if vector_custom:
                _scales = vector_scales
                num_scales = _scales.numel() if isinstance(_scales, torch.Tensor) else len(_scales)
                msg_parts.append(f"vector (n_scales={num_scales})")
            print(f"[{model_name}] Using custom diffusion scales instead of dyadic for " + ", ".join(msg_parts) + ".")

        # Print confirmation of vector track ablation
        if getattr(config, 'ablate_vector_track', False):
            print(f"[{model_name}] Ablating vector track.")
        if getattr(config, 'ablate_scalar_track', False):
            print(f"[{model_name}] Ablating scalar track.")


def _load_pretrained_weights(
    model: nn.Module,
    pretrained_dir: str,
    *,
    verbosity: int = 0,
) -> None:
    """
    Load pretrained parameters into *model* from *pretrained_dir*.

    The directory is expected to contain a weight file - first match in
    this priority order: ``model.safetensors``, ``pytorch_model.bin``,
    ``model.bin``, ``model.pt``, ``pytorch_model.pt``.  If none of those are found, the first file with extension ``.safetensors``, ``.bin`` or
    ``.pt`` is used.

    Missing or unexpected keys are ignored (``strict=False``) so transfer
    learning across tasks with different output layers works out of the box.
    """
    from pathlib import Path

    weights_dir = Path(pretrained_dir).expanduser()
    if not weights_dir.exists():
        raise FileNotFoundError(
            f"Pretrained weights directory '{weights_dir}' does not exist."
        )

    # Preferred filenames (ordered)
    candidate_files = [
        'model.safetensors',
        'pytorch_model.bin',
        'model.bin',
        'model.pt',
        'pytorch_model.pt',
    ]

    weight_file = None
    for fname in candidate_files:
        fp = weights_dir / fname
        if fp.is_file():
            weight_file = fp
            break

    # Fallback: first recognised extension
    if weight_file is None:
        for fp in weights_dir.iterdir():
            if fp.suffix in {'.safetensors', '.bin', '.pt'} and fp.is_file():
                weight_file = fp
                break

    if weight_file is None:
        raise FileNotFoundError(
            f"No weight file (.safetensors | .bin | .pt) found in '{weights_dir}'."
        )

    if verbosity > 0:
        print(f"[prep_model] Loading pretrained weights from {weight_file}")

    # Load state dict depending on file type
    if weight_file.suffix == '.safetensors':
        try:
            from safetensors.torch import load_file as safe_load_file
            state_dict = safe_load_file(str(weight_file))
        except ImportError as e:
            raise ImportError(
                "safetensors package is required to load .safetensors files. "
                "Install via 'pip install safetensors'."
            ) from e
    else:
        state_dict = torch.load(str(weight_file), map_location='cpu')

    # Strip 'module.' prefixes if model was saved under DDP
    cleaned_state = {
        (k[7:] if k.startswith('module.') else k): v
        for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)

    if verbosity > 0:
        print(
            f"[prep_model] Pretrained weights loaded. "
            f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
        )


def load_pretrained_weights_if_specified(
    model: nn.Module, 
    config: TrainingConfig
) -> None:
    """Load pretrained weights if specified in config.
    
    Args:
        model: The model to load weights into
        config: TrainingConfig object
    """
    if getattr(config, 'pretrained_weights_dir', None):
        _load_pretrained_weights(
            model,
            config.pretrained_weights_dir,
            verbosity=config.verbosity,
        )


# ============================================================
#               ESCGNN MODEL PREPARATION FUNCTIONS             
# ============================================================
def _prepare_escgnn_standard_model(
    config: TrainingConfig,
    dataloader_dict: Dict[str, PyGDataLoader],
    acc: Optional[Accelerator] = None
) -> Tuple[
    TrainingConfig, 
    nn.Module, 
    Dict[str, PyGDataLoader | DataLoader]
]:
    """
    Prepare a standard ESCGNN model and potentially modify dataloaders based on model mode.

    If model_mode == 'handcrafted_scattering', the dataloaders are modified to
    contain tensors of scattering features and targets, rather than PyTorch Geometric
    Data objects. The config is modified to reflect the change in dataloader_dict.

    Args:
        config: TrainingConfig object containing model and training parameters
        dataloader_dict: Dictionary of DataLoaders for each dataset split
        acc: Optional Accelerator object for distributed training
        
    Returns:
        Tuple of (config, model, _possibly_modified_dataloader_dict)
    """
    # Get model configuration
    model_config = config.model_config
    dataset_config = config.dataset_config
    
    # Set up shared parameters using helper functions
    base_module_kwargs = _setup_base_module_kwargs(
        acc.device, 
        config, 
        dataset_config
    )
    mlp_kwargs = _setup_mlp_kwargs(model_config, dataset_config)
    ablate_vector_track, ablate_scalar_track = _setup_ablation_flags(
        config, 
        dataset_config
    )

    # -------------------------------------------------------------
    # Set per-track parameter dicts (needed before applying nonlin selections)
    # -------------------------------------------------------------
    scalar_track_kwargs = {
        'feature_key': dataset_config.scalar_feat_key,
        'diffusion_op_key': dataset_config.scalar_operator_key,
        'num_layers': model_config.num_scattering_layers_scalar,
        'filter_combos_out': model_config.filter_combos_out,
        'diffusion_kwargs': {
            'scales_type': model_config.wavelet_scales_type,
            'J': model_config.J_scalar,
            'include_lowpass': True,
        },
        'J_prime': model_config.J_prime_scalar,
        'scattering_pooling_kwargs': {
            'pooling_type': model_config.pooling_type,
            'moments': model_config.moments,
            'nan_replace_value': model_config.nan_replace_value,
        },
        'diffusion_scales': model_config.infogain_scales_scalar
    }
    
    vector_track_kwargs = {
        'original_feature_key': dataset_config.vector_feat_key,
        'feature_key': dataset_config.vector_feat_key,
        'diffusion_op_key': model_config.vector_operator_key,
        'vector_dim': dataset_config.vector_feat_dim,
        'num_layers': 0 if ablate_vector_track else model_config.num_scattering_layers_vector,
        'filter_combos_out': model_config.filter_combos_out,
        'diffusion_kwargs': {
            'scales_type': model_config.wavelet_scales_type,
            'J': model_config.J_vector,
            'include_lowpass': True,
        },
        'J_prime': model_config.J_prime_vector,
        'scattering_pooling_kwargs': {
            'pooling_type': model_config.pooling_type,
            'moments': model_config.moments,
            'nan_replace_value': model_config.nan_replace_value,
            'norm_p': model_config.vector_norm_p,
        },
        'diffusion_scales': model_config.infogain_scales_vector
    }

    # --------------------------------------------------
    # Apply non-linearity selections *after* kwargs exist
    # --------------------------------------------------
    chosen_scalar = getattr(model_config, 'scalar_nonlin', None)
    chosen_mlp = getattr(model_config, 'mlp_nonlin', None)
    chosen_vector_nonlin = getattr(model_config, 'vector_nonlin', None)

    if chosen_mlp in MLP_NONLIN_MODULE_MAP:
        mlp_kwargs['nonlin_fn'] = MLP_NONLIN_MODULE_MAP[chosen_mlp]
        mlp_kwargs['nonlin_fn_kwargs'] = {}

    if chosen_scalar in SCALAR_NONLIN_FN_MAP:
        scalar_track_kwargs['nonlin_fn'] = SCALAR_NONLIN_FN_MAP[chosen_scalar]
        scalar_track_kwargs['nonlin_fn_kwargs'] = {}

    if chosen_vector_nonlin:
        vector_track_kwargs['vector_nonlin_type'] = chosen_vector_nonlin

    # Only used in cross-track mode
    cross_track_kwargs = {
        'n_cross_track_combos': getattr(model_config, 'n_cross_track_combos', 16),
        'n_cross_filter_combos': getattr(model_config, 'n_cross_filter_combos', 8),
        'within_track_combine': getattr(model_config, 'within_track_combine', False),
        'cross_track_mlp_hidden_dim': getattr(model_config, 'cross_track_mlp_hidden_dim', None),
        # Wavelet recombination parameters
        'use_wavelet_recombination': getattr(model_config, 'use_wavelet_recombination', True),
        'scalar_recombination_channels': getattr(model_config, 'scalar_recombination_channels', 16),
        'vector_recombination_channels': getattr(model_config, 'vector_recombination_channels', 16),
        'recombination_hidden_dim': getattr(model_config, 'recombination_hidden_dim', 64),
        'vector_gate_hidden_dim': getattr(model_config, 'vector_gate_hidden_dim', None),
        # Wjxs batch normalization parameter - default to True
        'use_wjxs_batch_norm': True,
    }
    
    # -------------------------------------------------------------
    # Handle custom diffusion scales specified in the dataset config
    # -------------------------------------------------------------
    scalar_track_kwargs['diffusion_scales'] = _process_scales(
        dataset_config.scalar_diffusion_scales
    )
    if not ablate_vector_track:
        vector_track_kwargs['diffusion_scales'] = _process_scales(
            dataset_config.vector_diffusion_scales
        )

    # Print model settings confirmations (only once on main process)
    scalar_custom = scalar_track_kwargs['diffusion_scales'] is not None
    vector_custom = vector_track_kwargs['diffusion_scales'] is not None
    _print_model_settings(
        config, scalar_custom, vector_custom,
        scalar_track_kwargs['diffusion_scales'], vector_track_kwargs['diffusion_scales'],
        "ESCGNN", acc
    )

    model = ESCGNN(
        mode=model_config.model_mode,
        ablate_vector_track=ablate_vector_track,
        ablate_scalar_track=ablate_scalar_track,
        stream_parallelize_tracks=config.use_cuda_streams \
            and torch.cuda.is_available(),
        base_module_kwargs=base_module_kwargs,
        mlp_kwargs=mlp_kwargs,
        scalar_track_kwargs=scalar_track_kwargs,
        vector_track_kwargs=vector_track_kwargs,
        cross_track_kwargs=cross_track_kwargs,
        verbosity=config.verbosity
    )

    # Optional: load pretrained weights (transfer learning)
    # _load_pretrained_weights_if_specified(model, config)
 
    # If in handcrafted_scattering mode, compute scattering features and create new dataloaders
    if model_config.model_mode == 'handcrafted_scattering':

        if acc.is_main_process:
            print(f"\nComputing scattering features...")

        # Compute scattering features and contain in new dataloaders dict
        scattering_features = {set_name: [] for set_name in dataloader_dict.keys()}
        targets = {set_name: [] for set_name in dataloader_dict.keys()}
        
        # Process each dataloader's data into tensors of scattering features and targets
        for set_name, dataloader in dataloader_dict.items():
            if acc.is_main_process:
                print(f"   Processing {set_name} set...")
            set_start_time = time.time()
            batch_times = []
            
            # Ensure model is in eval mode for feature computation
            model.eval()
            
            with torch.no_grad():
                for batch_i, batch in enumerate(dataloader):
                    batch_start_time = time.time()
                    
                    # Move batch to correct device
                    if acc is not None:
                        batch = batch.to(acc.device)
                    else:
                        batch = batch.to(config.device)
                    
                    # In handcrafted_scattering mode, model returns tensor of scattering features
                    features = model(batch)
                    target = batch.y
                    
                    # Move tensors to CPU for storage
                    features = features.cpu()
                    target = target.cpu()
                    
                    scattering_features[set_name].append(features)
                    targets[set_name].append(target)
                    
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
            
            # Synchronize processes before concatenating tensors
            if acc is not None:
                acc.wait_for_everyone()
            
            # Concatenate tensors
            scattering_features[set_name] = torch.cat(scattering_features[set_name], dim=0)
            targets[set_name] = torch.cat(targets[set_name], dim=0)
            
            # Synchronize processes after concatenation
            if acc is not None:
                acc.wait_for_everyone()
            
            if acc.is_main_process:
                set_time = time.time() - set_start_time
                print(f"      Complete.")
                print(f"      Total time for {set_name} set: {set_time:.2f}s")
                print(f"      Average batch time: {sum(batch_times) / len(batch_times):.2f}s")
        
        # Create new datasets and dataloaders
        scat_feats_dataset_dict = {
            set_name: TensorDataset(scattering_features[set_name], targets[set_name])
            for set_name in dataloader_dict.keys()
        }
        
        # Create new dataloaders with the same batch size and other parameters
        dataloader_dict = {
            set_name: DataLoader(
                dataset=dataset,
                batch_size=dataloader_dict[set_name].batch_size,
                shuffle=('train' in set_name),
                num_workers=dataloader_dict[set_name].num_workers,
                pin_memory=dataloader_dict[set_name].pin_memory,
                drop_last=dataloader_dict[set_name].drop_last
            ) for set_name, dataset in scat_feats_dataset_dict.items()
        }
        
        # Update config to use non-PyTorch Geometric data
        config.using_pytorch_geo = False
        config.target_key = 1  # Targets are second item in TensorDataset tuples
        
        # Create new MLP model for classification/regression
        model = VanillaNN(
            input_dim=scat_feats_dataset_dict['train'][0][0].shape[0],
            **mlp_kwargs,
            base_module_kwargs=base_module_kwargs
        )
        
        # Move model to correct device
        if acc is not None:
            model = model.to(acc.device)
        else:
            model = model.to(config.device)
    
    return config, model, dataloader_dict


def _prepare_escgnn_radial_model(
    config: TrainingConfig,
    dataloader_dict: Dict[str, PyGDataLoader],
    acc: Optional[Accelerator] = None
) -> Tuple[
    TrainingConfig, 
    nn.Module, 
    Dict[str, PyGDataLoader | DataLoader]
]:
    """
    Prepare a ESCGNNRadial model.

    ESCGNNRadial is a Bessel RBF version of ESCGNN that processes scalar 
    and vector tracks jointly via gated, distance-aware message passing.

    Args:
        config: TrainingConfig object containing model and training parameters
        dataloader_dict: Dictionary of DataLoaders for each dataset split 
            (just passed through to return value, for consistency with ESCGNN 
            standard model)
        acc: Optional Accelerator object for distributed training
        
    Returns:
        Tuple of (config, model, None [to match ESCGNN standard model's return])
    """
    # Get model configuration
    model_config = config.model_config
    dataset_config = config.dataset_config
    
    # Set up shared parameters using helper functions
    base_module_kwargs = _setup_base_module_kwargs(
        acc.device, 
        config, 
        dataset_config,
    )
    ablate_vector_track, ablate_scalar_track = _setup_ablation_flags(
        config, 
        dataset_config
    )

    # Get vector feature stats as dict of tensors (if not ablating vector 
    # track and its batch norm)
    if not ablate_vector_track \
    and not config.ablate_vector_wavelet_batch_norm:
        vec_feat_norms_stats = {
            'mean': torch.tensor(dataset_config.vector_norms_mean),
            'std': torch.tensor(dataset_config.vector_norms_std),
        }
        for v in vec_feat_norms_stats.values():
            # Dataset and its config is set upstream, before model is prepared,
            # but just in case, we check for None here.
            if v is None:
                raise ValueError("Vector feature stats are not set in the dataset config.")
    else:
        vec_feat_norms_stats = None


    # Handle custom diffusion scales specified in the dataset config
    custom_scalar_scales = _process_scales(model_config.scalar_diffusion_scales)
    custom_vector_scales = None if ablate_vector_track else _process_scales(model_config.vector_diffusion_scales)

    # Print model settings confirmations (only once on main process)
    scalar_custom = custom_scalar_scales is not None
    vector_custom = custom_vector_scales is not None
    _print_model_settings(
        config, scalar_custom, vector_custom,
        custom_scalar_scales, custom_vector_scales,
        "ESCGNNRadial", acc
    )

    # Map MLP nonlinearity
    mlp_nonlin_fn = nn.SiLU  # Default for ESCGNNRadial
    if hasattr(model_config, 'mlp_nonlin') and model_config.mlp_nonlin in MLP_NONLIN_MODULE_MAP:
        mlp_nonlin_fn = MLP_NONLIN_MODULE_MAP[model_config.mlp_nonlin]
    # Ensure we pass the class, not an instance
    if not isinstance(mlp_nonlin_fn, type):
        mlp_nonlin_fn = mlp_nonlin_fn.__class__

    # Check for required attributes and fail fast if missing
    if not hasattr(dataset_config, 'edge_rbf_key'):
        raise AttributeError("DatasetConfig is missing required attribute 'edge_rbf_key' for ESCGNNRadial.")
    if not hasattr(model_config, 'num_edge_features'):
        raise AttributeError("ModelConfig is missing required attribute 'num_edge_features' for ESCGNNRadial.")
    if not hasattr(dataset_config, 'num_atom_types'):
        raise AttributeError("DatasetConfig is missing required attribute 'num_atom_types' for ESCGNNRadial.")
    if not hasattr(model_config, 'gate_hidden_dim'):
        raise AttributeError("ModelConfig is missing required attribute 'gate_hidden_dim' for ESCGNNRadial.")

    model = ESCGNNRadial(
        # --- Data ---
        scalar_feature_key=dataset_config.scalar_feat_key,
        vector_feature_key=dataset_config.vector_feat_key,
        vec_feat_norms_stats=vec_feat_norms_stats,
        edge_rbf_key=dataset_config.edge_rbf_key,
        bond_type_key=dataset_config.bond_type_key,
        num_atom_types=dataset_config.num_atom_types,
        num_bond_types=dataset_config.num_bond_types,
        bond_emb_dim=model_config.edge_embedding_dim,
        num_rbf=model_config.num_edge_features,
        scalar_emb_dim=model_config.node_embedding_dim,
        d_vector=dataset_config.vector_feat_dim,
        # --- Scattering ---
        ablate_scalar_wavelet_batch_norm=config.ablate_scalar_wavelet_batch_norm,
        ablate_vector_wavelet_batch_norm=config.ablate_vector_wavelet_batch_norm,
        ablate_second_order_wavelets=config.ablate_second_order_wavelets,
        scalar_diffusion_op_key=model_config.scalar_operator_key,
        vector_diffusion_op_key=model_config.vector_operator_key,
        wavelet_J=model_config.J_scalar,
        wavelet_J_prime=model_config.J_prime_scalar,
        include_lowpass_wavelet=model_config.include_lowpass_wavelet,
        use_dirac_nodes=model_config.use_dirac_nodes,
        use_temporal_residuals=model_config.use_temporal_residuals,
        # --- Message passing ---
        num_msg_pass_layers=model_config.num_msg_pass_layers,
        use_residual_connections=model_config.use_residual_connections,
        # --- Scalar condensation MLP ---
        scalar_condense_hidden_dims=model_config.scalar_condense_hidden_dims,
        d_scalar_hidden=model_config.d_scalar_hidden,
        # --- Scalar gate MLP ---
        scalar_gate_mlp_hidden_dims=model_config.scalar_gate_mlp_hidden_dims,
        scalar_gate_nonlin_fn=MLP_NONLIN_MODULE_MAP[model_config.scalar_gate_nonlin],
        scalar_gate_rank=model_config.scalar_gate_rank,
        # --- Vector gate MLP ---
        vector_gate_mlp_hidden_dims=model_config.vector_gate_mlp_hidden_dims,
        vector_gate_nonlin_fn=MLP_NONLIN_MODULE_MAP[model_config.vector_gate_nonlin],
        vector_gate_rank=model_config.vector_gate_rank,
        # --- Pooling / readout ---
        pool_stats=model_config.pooling_type,
        readout_hidden_dims=model_config.mlp_hidden_dim,
        mlp_nonlin_fn=mlp_nonlin_fn,
        custom_scalar_scales=custom_scalar_scales,
        custom_vector_scales=custom_vector_scales,
        pred_output_dim=dataset_config.target_dim,
        base_module_kwargs=base_module_kwargs,
    )

    return config, model, dataloader_dict


def prepare_escgnn_model(
    config: TrainingConfig,
    dataloader_dict: Dict[str, PyGDataLoader],
    acc: Optional[Accelerator] = None
) -> Tuple[
    TrainingConfig, 
    nn.Module, 
    Dict[str, PyGDataLoader | DataLoader]
]:
    """
    Prepare a ESCGNN-family model (standard, radial, or modular) based on
    ``config.model_config.model_key``.

    This wrapper only handles ESCGNN variants. Comparison models are prepared by
    ``prepare_comparison_model``.

    Args:
        config: TrainingConfig object containing model and training parameters
        dataloader_dict: Dictionary of DataLoaders for each dataset split
        acc: Optional Accelerator object for distributed training
        
    Returns:
        Tuple of (config, model, _possibly_modified_dataloader_dict)
    """
    model_key = config.model_config.model_key
    
    if model_key == 'escgnn':
        return _prepare_escgnn_standard_model(config, dataloader_dict, acc)
    elif model_key == 'escgnn_radial':
        return _prepare_escgnn_radial_model(config, dataloader_dict, acc)
    elif model_key == 'escgnn_modular':
        return _prepare_escgnn_modular_model(config, dataloader_dict, acc)
    else:
        raise ValueError(
            f"Unsupported model_key for prepare_escgnn_model: {model_key}. "
            f"Supported values are 'escgnn', 'escgnn_radial', and 'escgnn_modular'."
        )


def _prepare_escgnn_modular_model(
    config: TrainingConfig,
    dataloader_dict: Dict[str, PyGDataLoader],
    acc: Optional[Accelerator] = None
) -> Tuple[
    TrainingConfig,
    nn.Module,
    Dict[str, PyGDataLoader | DataLoader]
]:
    """
    Prepare a ESCGNNModular model using explicit ModelConfig fields (no getattr fallbacks).
    """
    model_config = config.model_config
    dataset_config = config.dataset_config

    # Base module kwargs and ablations
    base_module_kwargs = _setup_base_module_kwargs(acc.device, config, dataset_config)
    ablate_vector_track, ablate_scalar_track = _setup_ablation_flags(config, dataset_config)
    ablate_second_order_wavelets = config.ablate_second_order_wavelets

    # Resolve effective scalar feature key: if scalar and vector keys collide
    # and Diracs are enabled, dataset prep routed concatenated [pos | dirac] to 'x'.
    # _scalar_key_cfg = dataset_config.scalar_feat_key
    # _vector_key_cfg = dataset_config.vector_feat_key
    # _use_diracs = getattr(model_config, 'use_dirac_nodes', False)
    # _effective_scalar_key = 'x' if (_use_diracs and (_scalar_key_cfg == _vector_key_cfg)) else _scalar_key_cfg

    # Track kwargs
    scalar_track_kwargs = {
        'feature_key': dataset_config.scalar_feat_key,
        'diffusion_op_key': model_config.scalar_operator_key,
        'diffusion_kwargs': {
            'scales_type': model_config.wavelet_scales_type,
            'J': model_config.J_scalar,
            'include_lowpass': model_config.include_lowpass_wavelet,
            'diffusion_scales': torch.tensor(model_config.scalar_diffusion_scales) \
                if model_config.scalar_diffusion_scales != 'dyadic' else 'dyadic',
        },
    }

    vector_track_kwargs = None
    if not ablate_vector_track:
        vector_track_kwargs = {
            'feature_key': dataset_config.vector_feat_key,
            'diffusion_op_key': model_config.vector_operator_key,
            'vector_dim': dataset_config.vector_feat_dim,
            'diffusion_kwargs': {
                'scales_type': model_config.wavelet_scales_type,
                'J': model_config.J_vector,
                'include_lowpass': model_config.include_lowpass_wavelet,
                'diffusion_scales': torch.tensor(model_config.vector_diffusion_scales) \
                    if model_config.vector_diffusion_scales != 'dyadic' else 'dyadic',
            },
        }

    # Mixing and neighbor kwargs from ModelConfig
    nonlin_map = MLP_NONLIN_MODULE_MAP
    mixing_kwargs = {
        'scalar_hidden_dims': model_config.scalar_wavelet_mlp_hidden,
        'vector_hidden_dims': model_config.vector_wavelet_mlp_hidden,
        'scalar_dropout_p': model_config.scalar_wavelet_mlp_dropout,
        'vector_dropout_p': model_config.vector_wavelet_mlp_dropout,
        'scalar_nonlin': nonlin_map.get(
            model_config.scalar_wavelet_mlp_nonlin,
            nn.SiLU
        ),
        'W_out_scalar': model_config.W_out_scalar,
        'W_out_vector': model_config.W_out_vector,
        'use_scalar_batch_norm': model_config.use_scalar_wavelet_batch_norm,
        'vector_distance_kernel': dataset_config.local_pca_distance_kernel_scale,
    }

    # Head kwargs are distinct from mixing kwargs; wire them explicitly
    head_kwargs = {
        'node_scalar_head_hidden': model_config.node_scalar_head_hidden,
        'node_scalar_head_nonlin': nonlin_map.get(model_config.node_scalar_head_nonlin, nn.SiLU),
        'node_scalar_head_dropout': model_config.node_scalar_head_dropout,
        'vector_gate_hidden': model_config.vector_gate_hidden,
        'vector_gate_nonlin': nonlin_map.get(model_config.vector_gate_nonlin, nn.SiLU),
        # Optional new gating behavior knobs (if present on ModelConfig)
        # Defaults are handled inside ESCGNNModular when keys are missing
        'vector_gate_use_sigmoid': model_config.vector_gate_use_sigmoid,
        'vector_gate_init_temperature': getattr(model_config, 'vector_gate_init_temperature', 1.0),
    }

    neighbor_kwargs = {
        'equal_degree': model_config.equal_degree,
        'k_neighbors': model_config.k_neighbors,
        'use_padding': model_config.neighbor_use_padding,
        'pool_stats': model_config.neighbor_pool_stats,
        'quantiles_stride': model_config.quantiles_stride
    }

    # Filter readout stats to supported ones ('mean','max','sum')
    _allowed_readout_stats = {'mean', 'max', 'sum'}
    _filtered_node_pool_stats = [s for s in model_config.neighbor_pool_stats if s in _allowed_readout_stats]
    if len(_filtered_node_pool_stats) == 0:
        _filtered_node_pool_stats = ['mean', 'max']

    readout_kwargs = {
        'type': 'agg' if model_config.equivar_pred else 'mlp',
        'mlp_hidden_dims': model_config.mlp_hidden_dim,
        'mlp_nonlin': model_config.mlp_nonlin,
        'node_pool_stats': _filtered_node_pool_stats,
    }

    model = ESCGNNModular(
        base_module_kwargs=base_module_kwargs,
        ablate_scalar_track=ablate_scalar_track,
        ablate_vector_track=ablate_vector_track,
        ablate_second_order_wavelets=ablate_second_order_wavelets,
        scalar_track_kwargs=scalar_track_kwargs,
        vector_track_kwargs=vector_track_kwargs,
        mixing_kwargs=mixing_kwargs,
        neighbor_kwargs=neighbor_kwargs,
        head_kwargs=head_kwargs,
        readout_kwargs=readout_kwargs,
    )

    return config, model, dataloader_dict


# ============================================================
#               COMPARISON MODEL PREPARATION                 
# ============================================================
def prepare_comparison_model(
    config: TrainingConfig,
    dataloader_dict: Dict[str, PyGDataLoader],
    acc: Optional[Accelerator] = None,
) -> Tuple[
    TrainingConfig,
    nn.Module,
    Dict[str, PyGDataLoader | DataLoader]
]:
    """
    Prepare a comparison model (EGNN, TFN, LEGS) by 
    wrapping it in a BaseModule-compatible adapter.
    """
    model_config = config.model_config
    model_key = model_config.model_key
    dataset_config = config.dataset_config
    task_lower = dataset_config.task.lower()
    print(f"_prepare_comparison_model: task_lower: {task_lower}")

    # BaseModule kwargs – ensure correct loss/metric setup
    base_module_kwargs = {
        'task': dataset_config.task,
        'metrics_kwargs': {
            'num_outputs': dataset_config.target_dim,
        },
        'verbosity': config.verbosity,
    }

    # Map loss function
    if hasattr(config, 'loss_fn') and config.loss_fn in LOSS_FN_MAP:
        base_module_kwargs['loss_fn'] = LOSS_FN_MAP[config.loss_fn]
        if config.loss_fn == 'huber':
            base_module_kwargs['loss_fn_kwargs'] = {
                'beta': getattr(config, 'huber_delta'),
                'reduction': 'mean',
            }
        else:
            base_module_kwargs['loss_fn_kwargs'] = {
                'reduction': 'mean',
            }

    # Instantiate underlying PyG model
    elif model_key == 'egnn':
        # If the dataset does not define atom types, pass in_dim=None to enable
        # EGNN's bias-initialized node features path
        inferred_in_dim = None
        if hasattr(dataset_config, 'num_atom_types'):
            try:
                nat = int(getattr(dataset_config, 'num_atom_types'))
                inferred_in_dim = nat if nat > 0 else None
            except Exception:
                inferred_in_dim = None
        pyg_model = EGNNModel(
            num_layers=model_config.comparison_model_num_layers,
            emb_dim=model_config.node_embedding_dim,
            in_dim=inferred_in_dim,
            out_dim=dataset_config.target_dim,
            # activation=model_config.mlp_nonlin,
            pool_types=model_config.pooling_type,
            # Follow config flag to control equivariant processing; the adapter/head will handle invariant targets.
            equivariant_pred=model_config.equivar_pred,
            predict_per_node=('node' in task_lower),
            vector_target=('vector' in task_lower),
        )
    elif model_key == 'tfn':
        # Infer in_dim similarly to EGNN: None -> bias path
        inferred_in_dim = None
        if hasattr(dataset_config, 'num_atom_types'):
            try:
                nat = int(getattr(dataset_config, 'num_atom_types'))
                inferred_in_dim = nat if nat > 0 else None
            except Exception:
                inferred_in_dim = None
        pyg_model = TFNModel(
            r_max=model_config.tfn_r_max,
            num_bessel=model_config.tfn_num_bessel,
            num_polynomial_cutoff=model_config.tfn_num_polynomial_cutoff,
            max_ell=model_config.tfn_max_ell,
            num_layers=model_config.comparison_model_num_layers,
            emb_dim=model_config.node_embedding_dim,
            hidden_irreps=None,
            mlp_dim=model_config.tfn_mlp_dim,
            in_dim=inferred_in_dim,
            out_dim=dataset_config.target_dim,
            aggr='sum',
            pool_types=model_config.pooling_type,
            gate=True,
            batch_norm=False,
            residual=True,
            # Follow config flag to control equivariant processing
            equivariant_pred=model_config.equivar_pred,
            predict_per_node=('node' in task_lower),
            vector_target=('vector' in task_lower),
            use_bias_if_no_atoms=True,
            radial_mode=model_config.tfn_radial_mode,
            radial_mlp_hidden=model_config.tfn_radial_mlp_hidden,
            radial_mlp_activation=model_config.tfn_radial_mlp_activation,
            unbiased_vector_pred_head=model_config.tfn_unbiased_vector_pred_head,
            radial_kernel_gaussian_eps=dataset_config.local_pca_distance_kernel_scale
,
        )
    elif model_key == 'legs':
        # Determine feature attribute and in_channels (using vector features as scalar channels)
        # scalar_key_cfg = dataset_config.scalar_feat_key
        # vector_key_cfg = dataset_config.vector_feat_key
        # # If scalar and vector keys collide (e.g., both 'pos'), we routed the
        # # composed scalar features to 'x' in dataset prep; use that here.
        # feat_key = 'x' \
        #     if (scalar_key_cfg == vector_key_cfg) and (model_config.use_dirac_nodes) \
        #     else scalar_key_cfg
        in_channels = dataset_config.vector_feat_dim
        if model_config.use_dirac_nodes:
            n_diracs = len(getattr(model_config, 'dirac_types', []) or [])
            in_channels += n_diracs

        pyg_model = LEGSModel(
            in_channels=in_channels,
            output_dim=dataset_config.target_dim,
            feature_attr=dataset_config.scalar_feat_key,
            J=model_config.J_scalar,
            # n_moments=model_config.moments[0] if hasattr(model_config, 'moments') else 4,
            # trainable_laziness=False,
            # apply_modulus_to_scatter=True,
            pool_types=model_config.pooling_type,
            predict_per_node=('node' in task_lower),
            mlp_head_hidden_dims=model_config.mlp_hidden_dim,
            node_mlp_dim=model_config.mlp_hidden_dim[0],
            activation=model_config.mlp_nonlin,
            mlp_dropout_p=model_config.mlp_dropout_p,
        )
    elif model_key in ('gcn', 'gin', 'gat'):
        # Shared setup for GCN/GIN/GAT backbones
        in_dim = 0
        if hasattr(dataset_config, 'num_atom_types'):
            try:
                nat = int(getattr(dataset_config, 'num_atom_types'))
                in_dim = nat if nat > 0 else getattr(dataset_config, 'vector_feat_dim', 0)
            except Exception:
                in_dim = getattr(dataset_config, 'vector_feat_dim', 0)
        else:
            in_dim = getattr(dataset_config, 'vector_feat_dim', 0)
        pyg_model = PyGModel(
            in_dim=in_dim,
            hidden_channels=model_config.node_embedding_dim or model_config.comparison_model_hidden_channels,
            out_dim=dataset_config.target_dim,
            num_layers=2 if model_config.comparison_model_num_layers is None else model_config.comparison_model_num_layers,
            feature_attr=dataset_config.scalar_feat_key,
            pool_types=model_config.pooling_type,
            predict_per_node=('node' in task_lower),
            mlp_head_hidden_dims=model_config.mlp_hidden_dim,
            mlp_dropout_p=model_config.mlp_dropout_p,
            activation=model_config.mlp_nonlin,
            backbone=model_key,
        )
    else:
        raise ValueError(f"Unsupported comparison model_key: {model_key}")

    # Wrap in ComparisonModel
    atomic_number_key = getattr(dataset_config, 'atomic_number_attrib_key', 'z')
    model = ComparisonModel(
        pyg_model=pyg_model,
        base_module_kwargs=base_module_kwargs,
        atomic_number_key=atomic_number_key,
    )

    # Move to device
    if acc is not None:
        model = model.to(acc.device)
    else:
        model = model.to(config.device)

    return config, model, dataloader_dict