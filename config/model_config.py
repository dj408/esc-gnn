# config/model_config.py
"""
This file contains the configuration for the model.
It is used to specify the model, the model architecture,
and the model initialization parameters.

Example yaml file:
model:
  model_key: 'escgnn'
  model_mode: 'handcrafted_scattering'
  wavelet_scales_type: 'dyadic'
  J: 4
  mlp_hidden_dim: [256, 128, 64, 32]
  mlp_dropout_p: 0.2  # Set to None for no dropout
"""
from dataclasses import dataclass, field
from typing import Optional, List, Any, Literal, Tuple, Union

@dataclass
class ModelConfig:
    
    # Model selection
    # Model types supported by the training pipeline.
    #  - 'escgnn_modular': A simplified, modular version of ESCGNN
    #  - 'mlp'  :  A simple multi-layer perceptron (used with handcrafted features)
    #  - 'egnn' | 'tfn' | 'legs': comparison baselines (wrapped)
    model_key: Literal[
        'escgnn_modular', 
        'mlp',
        'egnn', 
        'tfn', 
        'legs', 
        'gin',
        'gat'
    ] = 'escgnn'
    model_mode: Literal['handcrafted_scattering', 'filter-combine', 'cross-track'] = 'handcrafted_scattering'
    num_scattering_layers_scalar: int = 2
    num_scattering_layers_vector: int = 2
    column_normalize_P: bool = False
    equivar_pred: bool = True
    # Diffusion scales parameters (string or list of ints)
    # These allow custom diffusion scales to be specified for each track.
    # NOTE: these will override the model_config.p_wavelet_scales_type parameter,
    # If set to the string 'dyadic' (default), standard dyadic scales
    # controlled by the J parameter will be used. Otherwise, provide either
    # 1) a list of integers that will be shared across all channels, or
    # 2) a list of lists, giving per-channel custom scales.
    # The lists will be converted to torch.LongTensor objects later in the
    # model preparation step.
    scalar_diffusion_scales: Union[Literal['dyadic'], List[int], List[List[int]]] = 'dyadic'
    vector_diffusion_scales: Union[Literal['dyadic'], List[int], List[List[int]]] = 'dyadic'
    scalar_operator_key: str = 'P'
    vector_operator_key: str = 'Q'
    # flat_vector_feat_key: str = 'v_flat'  # legacy key for ESCGNN
    
    # ------------------------------------------------------------------
    # ESCGNN model architecture parameters
    # ------------------------------------------------------------------
    wavelet_scales_type: Literal['dyadic', 'infogain'] = 'dyadic'
    include_lowpass_wavelet: bool = True
    J_scalar: int = 4  # Number of wavelet scales for scalar track
    J_vector: int = 4  # Number of wavelet scales for vector track
    J_prime_scalar: Optional[int] = None  # Number of highest-frequency wavelets to keep in second layer (scalar)
    J_prime_vector: Optional[int] = None  # Number of highest-frequency wavelets to keep in second layer (vector)

    # ------------------------------------------------------------------
    # ESCGNN (v1-3) model parameters
    # ------------------------------------------------------------------
    # cross-track mode parameters
    n_cross_track_combos: int = 16
    n_cross_filter_combos: int = 8
    within_track_combine: bool = False  # If True, reuse per-track filter-combine layers inside cross-track mode

    # filter-combine mode parameters
    # (only used when model_mode == 'filter-combine')
    filter_combos_out: List[int] = field(
        default_factory=lambda: [16, 8]
    )
    
    # Cross-track per-wavelet MLP depth/width.  If None -> single Linear (legacy
    # behavior).  Otherwise the list of ints defines the hidden layer sizes
    # of an MLP applied *per wavelet* to mix scalar and vector channels before
    # cross-filter mixing.  Example: [128, 128] builds a 2-layer hidden MLP with
    # 128 units each.
    cross_track_mlp_hidden_dim: Optional[List[int]] = None

    # Wavelet recombination parameters (cross-track mode only)
    # Whether to use wavelet recombination layers in cross-track mode
    use_wavelet_recombination: bool = True
    
    # Number of output channels for scalar wavelet recombination
    scalar_recombination_channels: int = 16
    
    # Number of output channels for vector wavelet recombination  
    vector_recombination_channels: int = 16
    
    # Hidden dimension for recombination 2-layer MLPs
    recombination_hidden_dim: int = 64
    
    # Hidden dimension for vector gate MLP (defaults to recombination_hidden_dim if None)
    vector_gate_hidden_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # ESCGNNRadial-specific parameters
    # ------------------------------------------------------------------
    num_msg_pass_layers: int = 1
    use_residual_connections: bool = True
    
    # Hidden dimension for atom and bond type embeddings in ESCGNNRadial
    edge_embedding_dim: Optional[int] = None
    node_embedding_dim: Optional[int] = None

    # MLP hidden dims for scalar condensation
    scalar_condense_hidden_dims: List[int] = field(
        default_factory=lambda: [128, 128]
    )
    d_scalar_hidden: int = 64

    # MLP hidden dims for scalar gate
    scalar_gate_mlp_hidden_dims: List[int] = field(
        default_factory=lambda: [128, 128]
    )
    scalar_gate_nonlin: str = 'silu'
    scalar_gate_rank: int = 8

    # MLP hidden dims for vector gate
    vector_gate_mlp_hidden_dims: List[int] = field(
        default_factory=lambda: [128, 128]
    )
    vector_gate_nonlin: str = 'silu'
    vector_gate_rank: int = 8

    # Hidden dimension for gating MLPs in ESCGNNRadial (legacy, may be ignored)
    gate_hidden_dim: int = 128
    gate_rank: int = 8

    # Whether to use Dirac nodes in the scattering layers
    use_dirac_nodes: bool = False
    # Types of Dirac nodes to include as indicator channels, when enabled
    dirac_types: Optional[List[Literal['max', 'min']]] = field(
        default_factory=lambda: ['max', 'min']
    )
    use_temporal_residuals: bool = True

    # ------------------------------------------------------------------
    # InfoGain wavelet scales (optional)
    # ------------------------------------------------------------------
    # If provided, these override dyadic scales for the respective track.
    # Can be None (use dyadic), a 1D list/tensor (average scales for all channels),
    # or a 2D list/tensor (per-channel scales, shape [n_channels, n_scales])
    infogain_scales_scalar: Optional[Any] = None  # e.g. List[List[int]] or torch.Tensor
    infogain_scales_vector: Optional[Any] = None  # e.g. List[List[int]] or torch.Tensor

    # ------------------------------------------------------------------
    # Pooling parameters
    # ------------------------------------------------------------------
    pooling_type: Tuple[Literal['sum', 'max', 'median', 'statistical_moments'], ...] = ('sum')
    moments: Tuple[int, ...] = (1, 2, 3)  # Statistical moments to compute (1 = mean, 2 = variance, 3 = skewness, 4 = kurtosis)
    nan_replace_value: float = 0.0  # Value to replace NaN tensor values
    vector_norm_p: int = 2  # p-norm to use for vector features

    # ------------------------------------------------------------------
    # Nonlinearity choices
    # ------------------------------------------------------------------
    # Nonlinearity applied inside the ESCGNN scattering backbone (scalar track).
    scalar_nonlin: Literal['relu', 'silu', 'tanh'] = 'silu'

    # Nonlinearity applied between layers of the MLP head.
    #   Options correspond to torch.nn modules for easy instantiation in VanillaNN.
    mlp_nonlin: Literal['relu', 'silu', 'tanh'] = 'silu'

    # Nonlinearity used in the ESCGNN *vector track*.
    #   Options correspond to values accepted by ESCGNN.vector_track_kwargs['vector_nonlin_type'].
    #   Examples: 'shifted_relu', 'silu', 'silu-gate', 'relu-gate', 'sigmoid-gate', 'tanh', 'softplus'
    vector_nonlin: str = 'silu-gate'

    # MLP configuration
    mlp_hidden_dim: List[int] = field(
        default_factory=lambda: [256, 128, 64, 32]
    )
    mlp_dropout_p: Optional[float] = 0.2  # Set to None for no dropout
    mlp_use_batch_normalization: bool = False

    # ------------------------------------------------------------------
    # Comparison model hyperparameters
    # ------------------------------------------------------------------
    comparison_model_hidden_channels: int = 128  # Size of latent representation
    comparison_model_num_layers: int = 5
 
    # ------------------------------------------------------------------
    # Distance (e.g. Bessel) edge feature parameters
    # ------------------------------------------------------------------
    # Number of edge features (Bessel basis functions) per edge. If None, edge features are not computed or stored.
    num_edge_features: Optional[int] = 16

    # ------------------------------------------------------------------
    # TFN-specific defaults
    # ------------------------------------------------------------------
    tfn_r_max: float = 5.0
    tfn_num_bessel: int = 8
    tfn_num_polynomial_cutoff: int = 6
    tfn_max_ell: int = 2
    tfn_mlp_dim: int = 256
    # Radial embedding mode: also accepts kernel names ('gaussian','cosine_cutoff','epanechnikov')
    tfn_radial_mode: Literal['mlp_gates', 'bessel_cutoff', 'gaussian', 'cosine_cutoff', 'epanechnikov'] = 'mlp_gates'
    tfn_radial_mlp_hidden: List[int] = field(default_factory=lambda: [64, 64])
    tfn_radial_mlp_activation: Literal['relu', 'silu', 'swish', 'gelu'] = 'relu'
    # Equivariant vector head selection
    tfn_unbiased_vector_pred_head: bool = True

    # ------------------------------------------------------------------
    # ESCGNNModular-specific parameters
    # ------------------------------------------------------------------
    # Within-track wavelet-mixing MLPs
    scalar_wavelet_mlp_hidden: List[int] = field(default_factory=lambda: [32, 32])
    vector_wavelet_mlp_hidden: List[int] = field(default_factory=lambda: [32, 32])
    scalar_wavelet_mlp_dropout: float = 0.0
    vector_wavelet_mlp_dropout: float = 0.0
    scalar_wavelet_mlp_nonlin: Literal['relu', 'silu', 'tanh', 'gelu'] = 'silu'
    W_out_scalar: Optional[int] = 8
    W_out_vector: Optional[int] = 8
    use_scalar_wavelet_batch_norm: bool = True

    # Node head and vector gate MLPs
    node_scalar_head_hidden: List[int] = field(default_factory=lambda: [64, 64])
    node_scalar_head_nonlin: Literal['relu', 'silu', 'tanh', 'gelu'] = 'silu'
    node_scalar_head_dropout: float = 0.1
    vector_gate_hidden: List[int] = field(default_factory=lambda: [64, 64])
    vector_gate_nonlin: Literal['relu', 'silu', 'tanh', 'gelu'] = 'silu'
    vector_gate_use_sigmoid: bool = True
    vector_gate_init_temperature: float = 1.0
    use_scalar_in_vector_gate: bool = True
    use_neighbor_cosines: bool = True
    use_learned_static_vector_weights: bool = False

    # Neighbor cosine feature options
    equal_degree: bool = False
    k_neighbors: int = 5
    neighbor_use_padding: bool = True
    neighbor_pool_stats: List[str] = field(default_factory=lambda: ['max', 'mean', 'var'])
    # Controls the number of quantile stats when 'percentiles'/'quantiles' is enabled (slow)
    quantiles_stride: float = 0.2

