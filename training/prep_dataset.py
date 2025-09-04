import os
import warnings
from sympy import Q
import torch
import time
import multiprocessing as mp
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from torch.utils.data import Subset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.datasets import QM9

from models.class_maps import ATOM_WT_TO_IDX_MAP
from models.escgnn_data_classes import ESCGNNData, ESCGNNDatasetHDF5
from config.dataset_config import DatasetConfig
from config.train_config import TrainingConfig
from data_processing.process_pyg_data import (
    attach_coo_tensor_attr_to_data,
    process_edge_features_to_node_features,
)
from data_processing.ellipsoid_data_classes import (
    EllipsoidDatasetLoader,
    get_ellipsoid_dataset_info,
)
from accelerate import Accelerator
# from models.escgnn import ESCGNNData, ESCGNNDatasetHDF5
import h5py


def load_dataset(
    config: TrainingConfig,
    subset_indices: Optional[List[int]] = None,
    model_key: Optional[str] = None,
    print_info: bool = False,
) -> Data:
    """
    Load the dataset (on the main process) using the specified dataset class and parameters.
    If config.h5_path is set, return a ESCGNNDatasetHDF5 instance with operator keys from config.
    If config.h5_path and config.subsample_n are set, only use those indices present in the HDF5 file, 
    and subsample up to subsample_n.
    If config.data_subset_n is set, it overrides config.subsample_n and only uses that many samples.
    The subsample_seed is used for reproducibility of the subsampling.
    
    Note: In DDP mode, only the main process (rank 0) loads and prepares the dataset,
    but CPU workers are still used for parallel processing of operations like
    attaching diffusion operators.

    Args:
        config: DatasetConfig object containing dataset parameters
        subset_indices: Optional list of indices to subset the dataset to
        model_key: Optional model key
        print_info: Whether to print dataset information

    Returns:
        Dataset object with subset indices applied
    """
    data_config = config.dataset_config
    # Helper function for main process printing
    def main_print(*args, timestamp=False, indent=0, **kwargs):
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if timestamp:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}]", "  " * indent, *args, **kwargs)
            else:
                print("  " * indent, *args, **kwargs)
    
    # Start overall data preparation timing
    prep_start_time = time.time()
    main_print(f"\n{'='*60}")
    main_print(f"STARTING DATA PREPARATION", timestamp=True)

    dataset_kwargs = data_config.dataset_kwargs
    
    # Get number of CPU workers to use
    num_workers = data_config.num_workers if hasattr(data_config, 'num_workers') else 1
    main_print(f"Using {num_workers} CPU workers.", indent=1)

    # Handle dataset subsampling first
    if subset_indices is not None:
        # Explicit subset list takes precedence
        subset_size = None
    else:
        subset_size = getattr(data_config, 'debug_subset_n', None)
        if subset_size is None:
            subset_size = data_config.subsample_n
    
            # Initialize dataset with specified parameters
        if data_config.dataset.lower() == 'qm9':
            """
            QM9 
            
            PyTorch Geometric's version of the QM9 dataset contains 
            130,831 molecules with 11 features each. The features are:
            - atom type (H,C,N,O,F) [5 one-hot encoded cols]
            - atomic weight
            - hybridization type
            - aromatic (True/False)
            - mass
            - number of hydrogens (attached to non-hydrogen atoms)

            Anderson et al. (2019) ["Cormorant"] use a version of the QM9 dataset 
            that adds a twelfth property, thermochemical energy. They also create 
            principled train/test/valid splits (also used by E-GNN), which
            excludes 3054 molecules (from 133,885).

            E-GNN seems to use only the first 5 features (atom type) plus the 
            vector (position) feature, and not the bond type edge features.

            Atomic weight (categorical atom type) is stored in the 'z' attribute.
            """
            # edge_attr is a tensor of shape (num_edges, 4), with one-hot encoding
            # of the bond type [single, double, triple, aromatic]
            dataset_class = QM9
            def prep_qm9_feats_for_embeddings(data):
                # for nn.Embedding, we need consecutive indices from 0 to num_classes-1
                if config.model_config.node_embedding_dim is not None:
                    data.z = torch.tensor(
                        [ATOM_WT_TO_IDX_MAP[z] for z in data.z],
                        dtype=torch.int64
                    )
                if config.model_config.edge_embedding_dim is not None:
                    data.edge_type = data.edge_attr.argmax(dim=1)
                return data
            dataset_kwargs['transform'] = prep_qm9_feats_for_embeddings

        elif data_config.dataset.lower() == 'ellipsoids':
            """
            Ellipsoid datasets for testing equivariant vs non-equivariant GNNs.
            
            These datasets contain point clouds sampled from 3D ellipsoids with
            different coordinate biases:
            - Training set: ellipsoids biased toward having diameter along one axis
            - Test set: ellipsoids biased toward having diameter along a different axis
            
            An equivariant model should generalize well, while non-equivariant models
            that treat coordinates as separate scalar features should fail.
            
            The dataset contains:
            - x: Node features (xyz coordinates)
            - pos: Node positions (same as features for this dataset)
            - edge_index: k-NN graph edges
            - y: Target values (ellipsoid diameters)
            """
            # For ellipsoids, we'll load the dataset directly using our custom loader
            # The dataset_class will be set to None and handled specially below
            dataset_class = None
            dataset_kwargs['transform'] = None
        else:
            raise ValueError(f"Dataset '{data_config.dataset}' not supported")
    
    # Ensure dataset_class is defined irrespective of subsampling branch
    if 'dataset_class' not in locals():
        if data_config.dataset.lower() == 'qm9':
            dataset_class = QM9
            def prep_qm9_feats_for_embeddings(data):
                if config.model_config.node_embedding_dim is not None:
                    data.z = torch.tensor(
                        [ATOM_WT_TO_IDX_MAP[z] for z in data.z],
                        dtype=torch.int64
                    )
                if config.model_config.edge_embedding_dim is not None:
                    data.edge_type = data.edge_attr.argmax(dim=1)
                return data
            dataset_kwargs['transform'] = prep_qm9_feats_for_embeddings
        elif data_config.dataset.lower() == 'ellipsoids':
            dataset_class = None
            dataset_kwargs['transform'] = None
        else:
            raise ValueError(f"Dataset '{data_config.dataset}' not supported")

    # Load dataset
    main_print(f"Loading dataset...", indent=1)
    load_start_time = time.time()
    
    if dataset_class is not None:
        # Standard dataset loading (e.g., QM9)
        dataset = dataset_class(**dataset_kwargs)
        full_dataset_size = len(dataset)
    else:
        # Special handling for ellipsoid datasets
        if data_config.dataset.lower() == 'ellipsoids':
            # Get the data directory from config
            data_dir = getattr(data_config, 'data_dir', None)
            if data_dir is None:
                raise ValueError("data_dir must be specified in config for ellipsoid datasets")
            
            # THIS IS BROKEN: Print dataset information
            if print_info:
                info = get_ellipsoid_dataset_info(
                    data_dir, 
                    data_config.dataset_filename,
                    scalar_feat_key=data_config.scalar_feat_key,
                    vector_feat_key=data_config.vector_feat_key,
                    target_key=data_config.target_key,
                )
                main_print(f"Ellipsoid dataset info:", indent=2)
                main_print(f"  Dataset size: {info.get('dataset_size', 'N/A')}", indent=2)
                main_print(f"  Nodes per sample: {info.get('num_nodes_per_sample', 'N/A')}", indent=2)
                main_print(f"  Vector dimension: {info.get('vector_dim', 'N/A')}", indent=2)
                # The pickled dataset always includes graph-level 'y' (diameter).
                # Print it explicitly to avoid confusion with configured targets.
                if 'target_dim' in info:
                    main_print(f"  Graph target dimension: {info['target_dim']}", indent=2)
                # Also report the configured target selection for clarity
                main_print(
                    f"  Configured task/target: {data_config.task} / {data_config.target_key}"
                    f" (dim={data_config.target_dim})",
                    indent=2,
                )
                if 'target_mean' in info:
                    mean_print = info['target_mean']
                    if isinstance(mean_print, torch.Tensor):
                        if mean_print.numel() == 1:
                            mean_print = mean_print.item()
                        else:
                            mean_print = mean_print.squeeze().tolist()
                            mean_print = f"[{', '.join(f'{x.item():.3f}' for x in mean_print)}]"
                    main_print(f"  Target mean (first 100 samples): {mean_print:.3f}", indent=2)
                    std_print = info['target_std']
                    if isinstance(std_print, torch.Tensor):
                        if std_print.numel() == 1:
                            std_print = std_print.item()
                        else:
                            std_print = std_print.squeeze().tolist()
                            std_print = f"[{', '.join(f'{x.item():.3f}' for x in std_print)}]"
                    main_print(f"  Target std (first 100 samples): {std_print:.3f}", indent=2)
            
            # Load the single ellipsoid dataset
            dataset = EllipsoidDatasetLoader(
                data_dir=data_dir,
                dataset_filename=data_config.dataset_filename,
            )
            main_print(f"Loaded {len(dataset)} samples from ellipsoid dataset", indent=2)
        else:
            raise ValueError(f"Unknown dataset type: {data_config.dataset}")
    
    load_elapsed = time.time() - load_start_time
    load_min, load_sec = int(load_elapsed // 60), load_elapsed % 60
    main_print(f"Dataset loaded in {load_min}m {load_sec:.2f}s", indent=2)
    
    # ---------------------------------------------------------
    # Apply subsetting
    # ---------------------------------------------------------
    # Priority: by explicit indices
    if subset_indices is not None:
        main_print(f"Subsetting dataset to provided indices (n={len(subset_indices)})...", indent=1)
        dataset = Subset(dataset, subset_indices)

    # Fallback: random subsetting by size
    elif subset_size is not None and not getattr(data_config, 'h5_path', None):
        # Only perform early random subsampling when NOT using HDF5, because
        # HDF5-based runs must sample *after* checking available indices.
        main_print(f"Subsetting dataset...", indent=1)
        subset_start_time = time.time()

        # Use subsample_seed if available, otherwise use split_seed
        seed = data_config.subsample_seed if data_config.subsample_seed is not None else data_config.split_seed
        rng = np.random.default_rng(seed)

        # Generate subset indices
        total_size = len(dataset)
        subset_indices = rng.choice(
            total_size, 
            size=subset_size, 
            replace=False
        )

        dataset = Subset(dataset, subset_indices)

        subset_elapsed = time.time() - subset_start_time
        subset_min, subset_sec = int(subset_elapsed // 60), subset_elapsed % 60
        main_print(f"Original dataset size: {total_size}", indent=2)
        main_print(f"Subset size: {len(dataset)}", indent=2)
        main_print(f"Subset created in {subset_min}m {subset_sec:.2f}s", indent=2)
    
    # ------------------------------------------------------------------
    # Ensure (subset) dataset indices match HDF5 availability
    # We assume if the user wants the full dataset, they have put all samples
    # in the HDF5 file.
    # ------------------------------------------------------------------
    if data_config.subsample_n is not None and getattr(data_config, 'h5_path', None):
        print(f"Checking HDF5 file for available indices (subsample_n={data_config.subsample_n})...")
        # Determine which operator keys we will load – needed for intersection check
        required_op_keys = []
        if not getattr(data_config, 'ablate_scalar_track', False):
            required_op_keys.append(getattr(data_config, 'scalar_operator_key', 'P'))
        if not getattr(data_config, 'ablate_vector_track', False):
            required_op_keys.append(getattr(data_config, 'vector_operator_key', 'Q'))

        with h5py.File(data_config.h5_path, 'r') as _h5f:
            if 'original_idx' in _h5f:
                # Newer format with dedicated original_idx group; assume presence across all ops
                _available = {int(k) for k in _h5f['original_idx'].keys()}
            else:
                # Take intersection of keys present in all required operator groups
                if not all(op in _h5f for op in required_op_keys):
                    raise RuntimeError(
                        "HDF5 file missing one of the required operator groups: "
                        f"{required_op_keys}."
                    )
                key_sets = [{int(k) for k in _h5f[op].keys()} for op in required_op_keys]
                _available = set.intersection(*key_sets)

        if len(_available) == 0:
            raise RuntimeError("No common indices found across required operator keys in HDF5 file.")

        available_set = set(_available)

        # Current dataset may be a Subset (or not).  Recover *original* indices
        if isinstance(dataset, Subset):
            current_orig_indices = list(dataset.indices)
        else:
            current_orig_indices = list(range(len(dataset)))

        # Keep only positions whose original_idx is in the HDF5 file
        keep_positions = [pos for pos, orig in enumerate(current_orig_indices) if orig in available_set]

        # Optional further subsampling: honor subsample_n *after* filtering
        if (getattr(data_config, 'subsample_n', None) is not None) \
        and (len(keep_positions) > data_config.subsample_n):
            rng = np.random.default_rng(
                data_config.subsample_seed if data_config.subsample_seed is not None else data_config.split_seed
            )
            keep_positions = sorted(rng.choice(keep_positions, size=data_config.subsample_n, replace=False))

        if len(keep_positions) < len(current_orig_indices):
            main_print(
                f"Filtering dataset from {len(current_orig_indices)} to {len(keep_positions)} samples with HDF5 data.",
                indent=1,
            )
            dataset = Subset(dataset, keep_positions)

    # ------------------------------------------------------------------
    # Lazy wrapper to attach some attributes (e.g., original_idx) 
    # on first access
    # ------------------------------------------------------------------
    class _LazyAttributeLoadingDataset(torch.utils.data.Dataset):
        """Wrap a dataset (possibly nested Subsets) and guarantee `original_idx`."""

        def __init__(self, base_ds):
            self.base_ds = base_ds

        # --------------------------------------------------
        # Helper – resolve original QM9 index through nested Subsets
        # --------------------------------------------------
        @staticmethod
        def _resolve_orig(ds, idx):
            # Recursively resolve original index through nested Subsets
            while isinstance(ds, Subset):
                idx = ds.indices[idx]
                ds = ds.dataset
            return idx

        def __len__(self):
            return len(self.base_ds)

        def __getitem__(self, idx):
            data = self.base_ds[idx]
            if not hasattr(data, 'original_idx'):
                data.original_idx = self._resolve_orig(self.base_ds, idx)
            return data

    dataset = _LazyAttributeLoadingDataset(dataset)
    
    # If using HDF5 for P and Q, return ESCGNNDatasetHDF5
    if hasattr(data_config, 'h5_path') and data_config.h5_path:
        main_print(f"Loading HDF5 data...", indent=1)
        h5_start_time = time.time()
        
        # Convert all to ESCGNNData for correct collating/batching
        dataset = [
            ESCGNNData(
                **dict(d),
                operator_keys=(
                    getattr(data_config, 'scalar_operator_key', 'P'), 
                    getattr(data_config, 'vector_operator_key', 'Q')
                )
            ) \
            if not isinstance(d, ESCGNNData) \
            else d for d in dataset
        ]
        
        # Create index map for ESCGNNDatasetHDF5
        # The index map should map from the new indices (0 to len(dataset)-1)
        # to the original indices in the HDF5 file
        index_map = {i: d.original_idx for i, d in enumerate(dataset)}
        
        h5_elapsed = time.time() - h5_start_time
        h5_min, h5_sec = int(h5_elapsed // 60), h5_elapsed % 60
        main_print(f"HDF5 data loaded in {h5_min}m {h5_sec:.2f}s", indent=2)
        
        # Determine which operators to include based on ablation flags
        op_keys = ()
        if not getattr(data_config, 'ablate_scalar_track', False):
            op_keys = op_keys + (getattr(data_config, 'scalar_operator_key', 'P'),)
        if not getattr(data_config, 'ablate_vector_track', False):
            op_keys = op_keys + (getattr(data_config, 'vector_operator_key', 'Q'),)
        # No longer saving large adjacency matrix; only P/Q are sparse.

        if len(op_keys) > 0:
            dataset = ESCGNNDatasetHDF5(
                data_list=dataset,
                h5_path=data_config.h5_path,
                sparse_operators_tup=op_keys,
                index_map=index_map,
                scalar_feat_key=data_config.scalar_feat_key,
                vector_feat_key=data_config.vector_feat_key,
                attach_on_access=True,
                attributes_to_drop=data_config.attributes_to_drop,
                num_edge_features=data_config.num_edge_features,
                num_bond_types=data_config.num_bond_types,
            )
        
        # Skip diffusion operator processing since we're using HDF5
        proc_dataset = dataset
    
    # ------------------------------------------------------------------
    # IF NOT USING HDF5: 
    # Attach diffusion operators (sparse COO tensors) if specified
    # This alternative is more memory and processing intensive up front
    # ------------------------------------------------------------------
    elif data_config.diffusion_tensor_data_dir:
        main_print(f"Loading diffusion operators...", indent=1)
        diff_start_time = time.time()
        
        dir_path = os.path.join(data_config.data_dir, data_config.diffusion_tensor_data_dir)
        tensor_keys = ()
        if not getattr(data_config, 'ablate_scalar_track', False):
            tensor_keys = tensor_keys + (data_config.scalar_operator_key,)
        if not getattr(data_config, 'ablate_vector_track', False):
            tensor_keys = tensor_keys + (data_config.vector_operator_key,)
        if data_config['graph_construction'] == 'distance_cutoff':
            # Load graph structure tensors
            tensor_keys = tensor_keys + (
                'edge_index',
                'edge_weight',
            )
            # Optional edge features
            if getattr(data_config.model_config, 'num_edge_features', None):
                tensor_keys = tensor_keys + ('edge_features',)

        if len(tensor_keys) > 0:
            proc_dataset = attach_coo_tensor_attr_to_data(
                dataset=dataset,
                dir_path=dir_path,
                tensor_keys=tensor_keys,
                num_workers=num_workers,
            )
        else:
            proc_dataset = dataset
        
        diff_elapsed = time.time() - diff_start_time
        diff_min, diff_sec = int(diff_elapsed // 60), diff_elapsed % 60
        main_print(f"Diffusion operators loaded in {diff_min}m {diff_sec:.2f}s", indent=2)
        main_print(f"Loaded operators for {len(proc_dataset)} Data objects", indent=2)
    else:
        proc_dataset = dataset

    # --------------------------------------------------------------
    # Add/concat dense Dirac indicator channels to scalar features
    # for ellipsoids when requested by model config.
    # --------------------------------------------------------------
    if (
        hasattr(config, 'model_config')
        and hasattr(config, 'dataset_config')
        and isinstance(config.dataset_config.dataset, str)
        and config.dataset_config.dataset.lower() == 'ellipsoids'
    ):

        class _AddDiracFeaturesDataset(torch.utils.data.Dataset):
            """
            Wrap a dataset and add/concat Dirac indicator channels to data.x.

            If `data.diracs` exists (dict[str,int]), use those indices; otherwise, compute
            indices from pos norms according to `dirac_types` (e.g., ['max','min']).
            """
            def __init__(
                self, 
                base_ds: torch.utils.data.Dataset, 
                dirac_types: Optional[List[str]] = ['max', 'min'],
                scalar_feat_key: str = 'x',
            ):
                self.base_ds = base_ds
                self.dirac_types = dirac_types

            def __len__(self):
                return len(self.base_ds)

            def __getitem__(self, idx):
                data = self.base_ds[idx]
                vec_feat_key = config.dataset_config.vector_feat_key
                
                try:
                    # If Dirac channels were already appended for this sample, skip.
                    if hasattr(data, '_dirac_appended') \
                    and getattr(data, '_dirac_appended'):
                        return data
                    if not hasattr(data, vec_feat_key) \
                    or getattr(data, vec_feat_key) is None:
                        return data
                    pos = getattr(data, vec_feat_key)
                    num_nodes = pos.shape[0]
                    device = pos.device

                    # Resolve indices per type
                    indices: List[int] = []
                    diracs_map = getattr(data, 'diracs', None)
                    if isinstance(diracs_map, dict):
                        for t in self.dirac_types:
                            if t in diracs_map:
                                indices.append(int(diracs_map[t]))
                            else:
                                norms = torch.norm(pos, dim=1)
                                if t == 'max':
                                    indices.append(int(torch.argmax(norms).item()))
                                elif t == 'min':
                                    indices.append(int(torch.argmin(norms).item()))
                    else:
                        norms = torch.norm(pos, dim=1)
                        for t in self.dirac_types:
                            if t == 'max':
                                indices.append(int(torch.argmax(norms).item()))
                            elif t == 'min':
                                indices.append(int(torch.argmin(norms).item()))

                    if len(indices) == 0:
                        return data

                    dirac_channels = torch.zeros(
                        num_nodes,
                        len(indices),
                        dtype=torch.float32,
                        device=device,
                    )
                    for col, node_idx in enumerate(indices):
                        if 0 <= node_idx < num_nodes:
                            dirac_channels[node_idx, col] = 1.0

                    # --- Assign scalar feature key ---
                    scalar_feat_key = config.dataset_config.scalar_feat_key
                    vector_feat_key = config.dataset_config.vector_feat_key

                    # !!! ALWAYS CONCATENATE DIRACS TO THE END OF THE SCALAR FEATURES !!!
                    # If using Diracs but no scalar features already exist, set Diracs as new scalar features
                    if not hasattr(data, scalar_feat_key) or getattr(data, scalar_feat_key) is None:
                        data[scalar_feat_key] = dirac_channels
                    # If the scalar feature points to the vector feature 
                    # (treated as scalar features), concatenate Diracs to the vector features
                    elif (scalar_feat_key == vector_feat_key):
                        data[scalar_feat_key] = torch.cat(
                            [getattr(data, scalar_feat_key), dirac_channels],
                            dim=1,
                        )
                    # else, (non-vector) scalar features already exist, concatenate Diracs to these
                    else:
                        data[scalar_feat_key] = torch.cat([data[scalar_feat_key], dirac_channels], dim=1)

                    # Mark to avoid re-appending Diracs on subsequent accesses
                    setattr(data, '_dirac_appended', True)

                    # If ablating the vector track, concatenate Diracs to vector features treated as scalars                
                    # if config.model_config.ablate_vector_track:
                    #     
                    #     data[scalar_feat_key] = torch.cat(
                    #         [getattr(data, vec_feat_key), dirac_channels],
                    #         dim=1,
                    #     )
                    
                except Exception:
                    return data

                return data

        if config.model_config.use_dirac_nodes:
            dirac_types = (
                config.model_config.dirac_types
                if getattr(config.model_config, 'dirac_types') is not None
                else ['max', 'min']
            )
            proc_dataset = _AddDiracFeaturesDataset(proc_dataset, dirac_types=dirac_types)
            print(f"Added Dirac nodes ({dirac_types}) as scalar features.")

    if len(dataset) > len(proc_dataset):
        warnings.warn(
            f"\nDiffusion operators were computed for only a subset of "
            f"the dataset; only this subset will be used!"
        )
    
    # ------------------------------------------------------------------
    # Ensure dataset_config.vector_feat_dim matches the actual data
    # NOTE:  will fail in models where the vector features are treated as scalar features
    # and other scalar features are present (such as Diracs)
    # ------------------------------------------------------------------
    # try:
    #     vec_key = getattr(data_config, 'vector_feat_key', 'pos')
    #     _probe = proc_dataset[0] if len(proc_dataset) > 0 else None
    #     if _probe is not None and hasattr(_probe, vec_key) and getattr(_probe, vec_key) is not None:
    #         vec = getattr(_probe, vec_key)
    #         if isinstance(vec, torch.Tensor) and vec.dim() == 2:
    #             data_config.vector_feat_dim = int(vec.shape[1])
    # except Exception:
    #     pass
    
    # Print total preparation time
    prep_elapsed = time.time() - prep_start_time
    prep_min, prep_sec = int(prep_elapsed // 60), prep_elapsed % 60
    main_print(f"Complete.")
    main_print(f"Total data preparation time: {prep_min}m {prep_sec:.2f}s")
    # main_print(f"{'='*60}")
    
    return proc_dataset


def create_dataloaders(
    dataset: Data,
    splits_dict: Dict[str, torch.Tensor],
    config: TrainingConfig | DatasetConfig,
) -> Tuple[Dict[str, PyGDataLoader | DataLoader], TrainingConfig]:
    """
    Create DataLoaders for each split of the dataset.
    
    Args:
        dataset: The dataset to create loaders for
        splits_dict: Dictionary mapping split names to indices
        config: TrainingConfig or DatasetConfig object containing dataloader parameters
        
    Returns:
        Tuple with (a) dictionary with DataLoaders keyed by split name ('train', 'valid', 
        'test') and (b) the updated config object.
    """
    # Resolve DataLoader parameters with TrainingConfig taking precedence over DatasetConfig
    def _resolve_attr(attr_name, default=None):
        """Return attr from TrainingConfig if available, else DatasetConfig attr, else default."""
        if hasattr(config, attr_name):
            return getattr(config, attr_name)
        # If `config` is a TrainingConfig, it contains `.dataset_config`
        if hasattr(config, 'dataset_config') and hasattr(config.dataset_config, attr_name):
            return getattr(config.dataset_config, attr_name)
        return default

    # Determine DataLoader parameters
    dataset_cfg = config.dataset_config
    batch_size_cfg = _resolve_attr('batch_size', 128)
    evaluation_batch_size_cfg = _resolve_attr('evaluation_batch_size', batch_size_cfg)
    drop_last_cfg = _resolve_attr('drop_last', False)
    num_workers_cfg = _resolve_attr('dataloader_num_workers', _resolve_attr('num_workers', 0))
    pin_memory_cfg = _resolve_attr('pin_memory', False)

    # Build DataLoader kwargs
    dataloader_kwargs = {
        'drop_last': drop_last_cfg,
        'num_workers': num_workers_cfg,
        'pin_memory': pin_memory_cfg,
    }

    # ------------------------------------------------------------------
    # Safety: If CUDA is available and start method is 'fork' (default on Linux),
    # avoid initializing CUDA in worker subprocesses by forcing num_workers=0.
    # Users can opt-in to multiprocessing with CUDA by setting 'spawn' early
    # in their entrypoint (before any CUDA usage):
    #   import torch.multiprocessing as mp; mp.set_start_method('spawn', force=True)
    # ------------------------------------------------------------------
    try:
        start_method = mp.get_start_method(allow_none=True)
    except Exception:
        start_method = None
    if torch.cuda.is_available() and (start_method is None or start_method == 'fork'):
        if dataloader_kwargs['num_workers'] != 0:
            warnings.warn(
                "CUDA + forked DataLoader workers detected; setting num_workers=0 to avoid CUDA re-init in subprocesses. "
                "To use workers with CUDA, set spawn start method early in your entrypoint.")
            dataloader_kwargs['num_workers'] = 0

    # ------------------------------------------------------------------
    # Safety: Disable `pin_memory` when samples contain sparse tensors
    # ------------------------------------------------------------------
    # Torch (up to 2.4.1) does not implement `.pin_memory()` for sparse
    # tensors, which leads to a runtime NotImplementedError when the
    # DataLoader attempts to pin the entire Data object.  ESCGNN datasets
    # attach sparse COO tensors `P` and `Q` to each sample, so we must
    # ensure pinning is disabled in that case.
    if dataloader_kwargs['pin_memory'] and len(dataset) > 0:
        try:
            _probe = dataset[0]
            if any(
                isinstance(getattr(_probe, _attr, None), torch.Tensor) and getattr(_probe, _attr).is_sparse
                for _attr in ('P', 'Q')
            ):
                warnings.warn(
                    "pin_memory=True is incompatible with sparse tensors; disabling pin_memory for DataLoader.")
                dataloader_kwargs['pin_memory'] = False
        except Exception:
            # Fail-safe: if probing fails, leave pin_memory as-is
            pass

    # --------------------------------------------------------------
    # Optional: rotate *only* the test split (e.g., for ellipsoid datasets)
    # --------------------------------------------------------------    
    do_rotate_test = False
    if 'ellipsoid' in getattr(dataset_cfg, 'dataset', '').lower():
        # Allow either explicit flag (rotate_test_set) or default behaviour of False
        do_rotate_test = getattr(dataset_cfg, 'rotate_test_set', True)

    # Helper dataset wrapper to lazily rotate coordinates
    if do_rotate_test:

        class _RotatedDataset(torch.utils.data.Dataset):
            """Wrap an existing Dataset/Subset and rotates `pos` and specified vector attributes.

            Rotation is applied on-the-fly in __getitem__, keeping memory usage low.
            Currently implements a fixed 90-deg. rotation about the z-axis so that an
            original bias along the x-axis is mapped to the y-axis.
            """
            # Rotation matrix for 90 deg. about the z–axis (x -> y)
            R_90deg_z = torch.tensor([
                [0.0, -1.0, 0.0],
                [1.0,  0.0, 0.0],
                [0.0,  0.0, 1.0],
            ], dtype=torch.float32)

            def __init__(
                self, 
                base_ds: torch.utils.data.Dataset,
                # rotate_x_attrib: bool = True,
                vector_attribs_to_rotate: Optional[List[str]] = None,
                R: Optional[torch.Tensor] = None,
            ):
                self.base_ds = base_ds
                # self.rotate_x_attrib = rotate_x_attrib
                # Default vector attributes to rotate (common ellipsoid targets)
                if vector_attribs_to_rotate is None:
                    self.vector_attribs_to_rotate = [
                        'pos',
                        'y_base_normals',
                        'y_global_harmonic_normals', 
                        'y_multiscale_harmonic_normals',
                        'y_random_harmonic_normals',
                        'y_spectral_vector_field'
                    ]
                else:
                    self.vector_attribs_to_rotate = vector_attribs_to_rotate
                
                self.R = R if (R is not None) else self.R_90deg_z

            def __len__(self):
                return len(self.base_ds)

            def __getitem__(self, idx):
                data = self.base_ds[idx]

                # If [the first 3 cols of] x also store coordinates, rotate them too
                # if hasattr(data, 'x') and self.rotate_x_attrib:
                #     if data.x.dim() == 2 and data.x.shape[1] >= 3:
                #         device = data.x.device
                #         R = self.R.to(device)
                #         data.x[:, :3] = data.x[:, :3] @ R.T

                # Rotate all specified vector attributes
                for attr_name in self.vector_attribs_to_rotate:
                    if hasattr(data, attr_name):
                        attr_tensor = getattr(data, attr_name)
                        if attr_tensor is None:
                            continue
                        # Vector attributes are (N, d). If d > R_dim, rotate only first R_dim columns
                        if attr_tensor.dim() == 2:
                            device = attr_tensor.device
                            R = self.R.to(device)
                            d_rot = R.shape[0]
                            if attr_tensor.shape[1] >= d_rot:
                                if attr_tensor.shape[1] == d_rot:
                                    rotated = attr_tensor @ R.T
                                else:
                                    head = attr_tensor[:, :d_rot]
                                    tail = attr_tensor[:, d_rot:]
                                    rotated = torch.cat([head @ R.T, tail], dim=1)
                                setattr(data, attr_name, rotated)
 
                # ----------------------------------------------------------
                # Re-compute diffusion operators P and Q for the *rotated*
                # sample so that they align with the new coordinates.
                # We only do this once per Data object (marked by flag).
                # ----------------------------------------------------------
                if not getattr(data, '_pq_recomputed', False):
                    from data_processing.process_pyg_data import process_pyg_data

                    vec_key = getattr(dataset_cfg, 'vector_feat_key', 'pos')
                    # process_pyg_data returns a new Data/ESCGNNData object
                    data = process_pyg_data(
                        data,
                        vector_feat_key=vec_key,
                        device=data[vec_key].device \
                            if hasattr(data, vec_key) else 'cpu',
                        return_data_object=True,
                        num_edge_features=getattr(config.model_config, 'num_edge_features'),
                        hdf5_tensor_dtype=getattr(config.dataset_config, 'hdf5_tensor_dtype', 'float16'),
                        graph_construction=None,
                        use_mean_recentering=getattr(config.dataset_config, 'use_mean_recentering'),
                        sing_vect_align_method=getattr(config.dataset_config, 'sing_vect_align_method', 'column_dot'),
                        local_pca_kernel_fn_kwargs={
                            'kernel': getattr(config.dataset_config, 'local_pca_distance_kernel', 'gaussian'),
                            'gaussian_eps': getattr(config.dataset_config, 'local_pca_distance_kernel_scale', None),
                        }
                    )
                    data._pq_recomputed = True

                return data

    # ------------------------------------------------------------------
    # Vector feature stats calculation (QM9 only, if needed)
    # ------------------------------------------------------------------
    if not config.ablate_vector_wavelet_batch_norm \
    and 'qm9' in dataset_cfg.dataset.lower():
        mean, std = get_train_set_vector_feat_norms_stats(
            dataset, dataset_cfg, splits_dict
        )
        dataset_cfg.vector_norms_mean = mean.tolist() if isinstance(mean, torch.Tensor) else mean
        dataset_cfg.vector_norms_std = std.tolist() if isinstance(std, torch.Tensor) else std
        print(f"Vector feature norms stats: mean = {mean}, std = {std}")
    
    # ------------------------------------------------------------------
    # Target preprocessing
    # ------------------------------------------------------------------
    # (1) Subset targets on all data in all splits, if specified
    target_incl_idx = dataset_cfg.target_include_indices
    if target_incl_idx is not None:
        for data in dataset:
            # print(f"data.y shape:\n\tbefore: {list(data.y.shape)}")
            data.y = data.y.squeeze()[target_incl_idx]
            # print(f"\tafter: {list(data.y.shape)}")

    # (2) Apply normalization in-place to *train data only*, if specified
    # (we want valid and test metrics to be computed with de-normalized
    # output, versus raw targets, to get metrics in the original units) 
    if (dataset_cfg.target_preprocessing_type is not None):
        stats_dict = getattr(dataset_cfg, 'target_preproc_stats', None)
        if stats_dict is not None:
            center = stats_dict.get('center', None)
            scale = stats_dict.get('scale', None)
        else:
            center, scale = None, None

        # (a) Compute preprocessing stats if not already present in config
        if (stats_dict is None) \
        or ((stats_dict is not None) and (center is None or scale is None)):
            print(f"Need to compute target preprocessing stats (not found in dataset config):")
            # (i) MAD normalization
            if dataset_cfg.target_preprocessing_type == 'mad_norm':
                print(f"\tComputing MAD normalization stats from train set targets...")
                # NOTE: Target subsetting has already been applied above, so stats
                # will be computed on the correct subset of targets
                center, scale = get_train_set_targets_means_mads(
                    dataset, splits_dict
                ) # shapes (d_target,), (d_target,)
                dataset_cfg.target_preproc_stats = {
                    'center': center.tolist() \
                        if isinstance(center, torch.Tensor) else center,
                    'scale': scale.tolist() \
                        if isinstance(scale, torch.Tensor) else scale
                }
            # (ii) TODO: add other preprocessing methods
            else: 
                raise ValueError(
                    f"Target preprocessing not implemented for '{dataset_cfg.target_preprocessing_type}'"
                )
            print("\tDone.")

        # (b) Rescale targets in train split
        print(f"Rescaling targets in train split...")
        # Ensure stats are tensors for correct broadcasting with tensor targets
        center_t = center if isinstance(center, torch.Tensor) \
            else torch.as_tensor(center)
        scale_t = scale if isinstance(scale, torch.Tensor) \
            else torch.as_tensor(scale)
        for i in splits_dict['train']:
            # Note: dataset[i].y has already been subset to only 'target_incl_idx' in step (1) above
            dataset[i].y = (dataset[i].y - center_t) / scale_t  # (N,)
        print("\tDone.")
    else:
        print(f"No target preprocessing applied.")

    # --------------------------------------------------------------
    # DataLoader selection and collation
    # --------------------------------------------------------------
    if len(dataset) > 0 and _resolve_attr('using_pytorch_geo', True):
        dataloader_kwargs['collate_fn'] = Batch.from_data_list
        dataloader_class = PyGDataLoader
    else:
        dataloader_class = DataLoader

    # ------------------------------------------------------------------
    # Helper dataset wrapper to compute Euclidean edge distances once
    # ------------------------------------------------------------------
    compute_edge_distances = getattr(dataset_cfg, 'compute_edge_distances', False)
    if compute_edge_distances:
        class _WithEdgeDistances(torch.utils.data.Dataset):
            def __init__(self, base_ds: torch.utils.data.Dataset, vec_key: str = 'pos'):
                self.base_ds = base_ds
                self.vec_key = vec_key

            def __len__(self):
                return len(self.base_ds)

            def __getitem__(self, i: int):
                d = self.base_ds[i]
                try:
                    if (not hasattr(d, 'edge_weight') or getattr(d, 'edge_weight') is None) \
                    and hasattr(d, 'edge_index') and hasattr(d, self.vec_key):
                        pos = getattr(d, self.vec_key)
                        ei = d.edge_index
                        src, dst = ei[0], ei[1]
                        vec = pos[src] - pos[dst]
                        dist = torch.norm(vec, dim=-1)
                        d.edge_weight = dist
                except Exception:
                    # Best-effort; leave sample unchanged on failure
                    return d
                return d

    # Create dataloaders for each split using Subset,
    # and apply rotation to test split if requested
    dataloader_dict = {}
    for set_name, idx in splits_dict.items():
        split_ds = Subset(dataset, idx)
        # Optionally compute Euclidean edge distances once (stored in edge_weight)
        if compute_edge_distances:
            split_ds = _WithEdgeDistances(split_ds, vec_key=getattr(dataset_cfg, 'vector_feat_key', 'pos'))
        # Apply rotation lazily only to test split if enabled
        if do_rotate_test and ('test' in set_name.lower()):
            # Get vector attributes to rotate from config if available
            vector_attribs_to_rotate = getattr(dataset_cfg, 'vector_attribs_to_rotate', None)
            split_ds = _RotatedDataset(
                split_ds, 
                vector_attribs_to_rotate=vector_attribs_to_rotate
            )
        # Choose batch size: train uses training batch size; valid/test use evaluation batch size
        lower_name = set_name.lower()
        if 'train' in lower_name:
            requested_bs = batch_size_cfg
        elif ('val' in lower_name) or ('test' in lower_name):
            requested_bs = evaluation_batch_size_cfg
        else:
            requested_bs = batch_size_cfg

        split_batch_size = min(len(split_ds), requested_bs)

        dataloader_dict[set_name] = dataloader_class(
            dataset=split_ds,
            shuffle=('train' in set_name),
            batch_size=split_batch_size,
            **dataloader_kwargs
        )
    
    return dataloader_dict, config


def get_train_set_targets_means_mads(
    dataset: Data,
    # dataset_cfg: DatasetConfig,
    splits_dict: Dict[str, torch.Tensor],
) -> Tuple[float, float]:
    """
    Compute mean (center) and MAD (scale) of (individual) targets on the 
    train split, if not already stored in the dataset config.
    
    Note: This function assumes that target subsetting has already been applied
    to the dataset if target_include_indices is specified.
    """
    # Don't need this guard since this function should be called only if needed
    # if (dataset_cfg.target_preprocessing_type == 'mad_norm') \
    # and ((dataset_cfg.target_preproc_stats is None) \
    #      or ('center' in dataset_cfg.target_preproc_stats \
    #          and dataset_cfg.target_preproc_stats['center'] is None) \
    #      or ('scale' in dataset_cfg.target_preproc_stats \
    #          and dataset_cfg.target_preproc_stats['scale'] is None) \
    #     ):

    train_idx = splits_dict.get('train', list(range(len(dataset))))
    if isinstance(train_idx, torch.Tensor):
        train_idx = train_idx.tolist()

    train_targets = torch.stack(
        [dataset[i].y.squeeze() for i in train_idx],
        dim=0
    ) # (N, d_target) - where d_target is the subset if target_include_indices was applied
    
    # Ensure we have the right shape for single-target case
    if train_targets.dim() == 1:
        train_targets = train_targets.unsqueeze(1)  # (N, 1)
    
    mean = torch.mean(train_targets, dim=0) # (d_target,)
    mad = torch.mean(torch.abs(train_targets - mean), dim=0) # (d_target,)
    mad = torch.where(mad < 1e-12, torch.ones_like(mad), mad)

    # For single target, return scalars instead of tensors
    if mean.numel() == 1:
        mean = mean.item()
        mad = mad.item()

    # print(f"Train set target stats:\n\tmean: {list(mean) if hasattr(mean, '__iter__') else mean}\n\tmad: {list(mad) if hasattr(mad, '__iter__') else mad}")

    return mean, mad


def get_train_set_vector_feat_norms_stats(
    dataset: Data,
    dataset_cfg: DatasetConfig,
    splits_dict: Dict[str, torch.Tensor],
) -> Tuple[float, float]:
    """
    Compute mean and std of vector feature norms on the train split,
    if not already stored in the dataset config.
    """
    if (dataset_cfg.vector_norms_mean is not None) \
    and (dataset_cfg.vector_norms_std is not None):
        mean, std = dataset_cfg.vector_norms_mean, dataset_cfg.vector_norms_std
    else:
        train_idx = splits_dict.get('train', list(range(len(dataset))))
        if isinstance(train_idx, torch.Tensor):
            train_idx = train_idx.tolist()

        vector_feat_key = dataset_cfg.vector_feat_key
        # Each graph's vec feat is (num_nodes_in_graph, d_vector) -> get (num_nodes_in_graph,) norms
        # -> stack all nodes across all graphs to get global norm stats
        vec_feat_norms = torch.cat([
            torch.norm(dataset[i][vector_feat_key], dim=1) for i in train_idx
        ], dim=0) # (N,)
        mean, std = torch.mean(vec_feat_norms), torch.std(vec_feat_norms)
    return mean, std
    

def get_torchmd_qm9_splits(
    len_dataset: int,
) -> Dict[str, torch.Tensor]:
    """
    Equiformer (arxiv.org/pdf/2206.11990, Section D.1) claims to have
    taken the its splits from TorchMD for the QM9 dataset. Equiformer uses
    the following code (streamlined here, with seed=1) to generate these splits:
    https://github.com/atomicarchitects/equiformer/blob/master/datasets/pyg/qm9.py#L181
    """
    SEED = 1  # intentionally hardcoded
    rng = np.random.default_rng(SEED)
    data_perm = rng.permutation(len_dataset)
    # TorchMD scheme expects full QM9 (~130k samples).  If the dataset is smaller
    # (e.g., due to subsampling or HDF5 availability), fall back to proportional
    # splits to avoid empty valid/test sets.
    n_train_full, n_valid_full = 110_000, 10_000

    if len_dataset > (n_train_full + n_valid_full):
        n_train, n_valid = n_train_full, n_valid_full
    else:
        # Proportional split
        n_train = int(0.8407793260007185 * len_dataset)
        n_valid = int(0.0764344841818835 * len_dataset)

    train, valid, test = np.split(
        data_perm,
        [n_train, (n_train + n_valid)]
    )
    splits_dict = {
        'train': train,
        'valid': valid,
        'test': test
    }
    return splits_dict

