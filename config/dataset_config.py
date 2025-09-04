# config/dataset_config.py
"""
This file contains the default/overridable configuration for the dataset.
It is used to specify the dataset, the split parameters,
and the data loader parameters.
"""

from dataclasses import dataclass
from typing import Optional, Any, Dict, Union, List, Callable, Literal
import torch
from torch_geometric.datasets import QM9

@dataclass
class DatasetConfig:
    """Configuration for dataset initialization and behavior.
    
    h5_path: Optional path to HDF5 file storing, e.g., P and Q tensors for each sample. 
    If set, dataset loading will not attach P and Q, but expects them to be loaded 
    from HDF5 in the custom Dataset class.
    subsample_n: If set, only use a random subset of this many samples (overridden by clarg if provided).
    subsample_seed: Random seed for dataset subsampling (used for both subsample_n and data_subset_n).
    target_include_indices: List of indices to keep from the target (y) tensor. If set, the y attribute in each Data/Batch object will be subset to include only these indices. Useful for selecting a subset of regression targets (e.g., for QM9).
    """
    
    # Required paths
    data_dir: str = None # must be provided by user in yaml file
    dataset_filename: Optional[str] = None
    diffusion_tensor_data_dir: Optional[str] = "pq_tensor_data"
    h5_path: Optional[str] = None  # Path to HDF5 file, e.g., for P and Q tensors
    subsample_n: Optional[int] = None  # If set, only use a random subset of this many samples
    subsample_seed: Optional[int] = 123456  # Random seed for dataset subsampling

    # Dataset selection and basic parameters
    dataset: str = 'QM9'
    scalar_feat_key: str = 'x'
    vector_feat_key: str = 'pos'
    atomic_number_attrib_key: str = 'z'
    bond_attr_key: str = 'edge_attr'
    bond_type_key: str = 'edge_type'
    num_bond_types: int = 4  # Number of bond types for embedding
    vector_feat_dim: int = 3
    task: str = 'graph_regression'
    target_key: str = 'y'
    target_dim: int = 1  # 19 available
    num_atom_types: int = 5  # Number of atom types for embedding (e.g., 5 for H,C,N,O,F in QM9)

    # Optional list of node feature indices to keep. If set, the x attribute of each Data/Batch object
    # will be subset to include ONLY these feature columns in the given order.
    node_feats_include_indices: Optional[List[int]] = None

    # Vector feature preprocessing
    vector_norms_mean: Optional[float] = None
    vector_norms_std: Optional[float] = None
    
    # Node C_i rank deficiency handling
    rank_deficiency_strategy: Optional[Literal['Tikhonov']] = None  # Strategy for handling k < d (e.g., 'Tikhonov')
    tikhonov_epsilon: float = 0.001  # Small isotropic regularisation strength if strategy == 'Tikhonov'
    # Alignment method for local PCA singular vectors across neighbors
    sing_vect_align_method: Literal['column_dot', 'procrustes'] = 'column_dot'

    # Local PCA distance kernel for weighting vector neighbors
    local_pca_distance_kernel: Literal[
        "gaussian", "cosine_cutoff", "epanechnikov",
    ] = 'cosine_cutoff'
    local_pca_distance_kernel_scale: Optional[float] = None
    use_mean_recentering: bool = False

    # Radial/continuous feature processing (don't confuse with categorical edge 'type' attributes)
    edge_rbf_key: str = 'edge_features'
    num_edge_features: int = 0

    # ------------------------------------------------------------------
    # Graph construction settings
    # ------------------------------------------------------------------
    # How to build the graph (edge_index) used for diffusion operators and
    # any downstream message-passing layers.
    #   - 'chemical_bonds'   – keep the bond graph shipped with QM9
    #   - 'distance_cutoff'  – connect any pair of atoms within `distance_cutoff` angstroms
    #                          (default and dataset-agnostic)
    graph_construction: Optional[Literal['k-nn', 'radius', 'reweight_existing_edges']] = None

    # Distance threshold for the radius-graph when `graph_construction == 'distance_cutoff'`
    distance_cutoff: float = 5.0

    # Cap on neighbours per node
    max_num_neighbors: int = 16

    # DEPRECATED: process edge features into node features
    use_edge_as_node_features: bool = False  # Whether to process edge features into node features
    edge_to_node_feature_key: str = 'bond_node_features'  # Key for the processed edge features

    # ------------------------------------------------------------------
    # HDF5 tensor precision
    # ------------------------------------------------------------------
    # Floating-point dtype used when saving P, Q, A and edge-feature tensors
    # to HDF5.  Accepts any torch dtype name understood by `torch.tensor`
    # constructor (e.g. 'float32', 'float16').  Default float16 halves the
    # storage requirement relative to float32 without significant loss for
    # QM-level datasets.
    hdf5_tensor_dtype: str = 'float16'

    # ------------------------------------------------------------------
    # Dataset split strategy
    # ------------------------------------------------------------------
    # For QM9, True (default) the data
    # preparation pipeline uses the deterministic TorchMD split scheme
    # adopted by Equiformer (110,000 training, 10,000 validation, remainder
    # test) via `prep_dataset.get_torchmd_qm9_splits`.  Set to False to
    # fall back to random proportional splits controlled by `train_prop`
    # and `valid_prop`.
    use_torchmd_qm9_splits: bool = True
    split_seed: int = 4489670
    # Number of folds for cross-validation (when experiment_type is 'kfold')
    k_folds: int = 5
    train_prop: float = 0.8
    valid_prop: float = 0.2

    # Other data preparation parameters
    force_reload: bool = False  # whether to force reload of cached, pre-transformed data

    # DataLoader parameters
    num_workers: int = 4
    # batch_size: int = 128  # overridden by training.batch_size
    drop_last: bool = False
    pin_memory: bool = True  # doesn't work with sparse tensors in older versions of PyTorch
    using_pytorch_geo: bool = True

    # --------------------------------------------------------------
    # Optional: remove unused attributes before batching to GPU
    # --------------------------------------------------------------
    # E.g. QM9 ships 'x', 'smiles', 'name' that ESCGNN does not need.
    # Specify a list of attribute names to delete during the custom
    # collate function (batch-wise attachment path).  Keeps host/GPU
    # memory lower and reduces serialization overhead.
    attributes_to_drop: Optional[List[str]] = None

    # Target subsetting
    target_include_indices: Optional[List[int]] = None  # Indices to include from the target (y) tensor

    # Target preprocessing
    # If 'mad_norm', each target property will be transformed as
    #   y_norm = (y - mean) / MAD,
    # where MAD = mean(|y - mean|). Stats are computed once on the
    # loaded dataset and stored in `target_preproc_stats`.
    target_preprocessing_type: Optional[Literal['mad_norm']] = None # 'None' means no preprocessing
    target_preproc_stats: Optional[Dict[str, Any]] = None  # {'mean': list, 'mad': list}

    # Ellipsoid-specific rotation settings
    rotate_test_set: bool = True  # Whether to rotate test set for equivariance testing
    vector_attribs_to_rotate: Optional[List[str]] = None  # List of vector attributes to rotate in test set

    # Whether to compute and store Euclidean edge distances into `edge_weight`
    # during dataset preparation when not already provided by the dataset.
    compute_edge_distances: bool = False

    
    def __post_init__(self):
        """Convert parameters to dataset kwargs."""
        self.dataset_kwargs = {}
        
        # Add dataset-specific parameters
        if self.dataset == 'QM9':
            self.dataset_kwargs.update({
                'root': self.data_dir,
                'force_reload': self.force_reload
            })
        elif self.dataset.lower() == 'ellipsoids':
            # For ellipsoids, we don't need additional kwargs as we load directly
            # But we can set some default parameters
            self.dataset_kwargs.update({
                'root': self.data_dir,  # Keep for compatibility
            })
            # Do not coerce target/task defaults here; respect YAML/CLI precedence
        # Add more dataset types here as needed

