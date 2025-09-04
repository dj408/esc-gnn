"""
Utility functions for pytorch-geometric
projects.
"""
import torch
from torch import linalg
from torch_geometric.data import (
    Data,
    Batch,
    Dataset,
    InMemoryDataset
)
from typing import (
    Tuple,
    List,
    Dict,
    Any,
    Optional,
    Iterable,
    Callable,
    Literal
)

def check_if_undirected(edge_index: torch.Tensor) -> bool:
    """
    Checks if an edge_index tensor is undirected (has two
    directions for each edge).

    Args:
        edge_index: edge_index tensor of shape (2, E).
    Returns:
        True if the edge_index is undirected, False otherwise.
    """
    edges = set(map(tuple, edge_index.t().tolist()))
    reverse_edges = set((j, i) for i, j in edges)
    return edges == reverse_edges


class GraphListDataset(Dataset):
    """
    This class is a simple PyG Dataset subclass useful 
    for converting a list of Data objects into a Dataset,
    and applying transformations on every Data (graph) 
    object access, if needed.

    __init__ args:
        data_list: list of pytorch-geometric Data objects.
        transform: a callable function to apply to each
            Data object on every access.
    """
    def __init__(
        self,
        data_list: List[Data],
        transform: Optional[Callable] = None):
        super().__init__(transform=transform)
        self.data_list = data_list  # Store the list of Data objects

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        # note self.transform will be applied right after
        # self.get, within self.__get_item parent call
        return self.data_list[idx]


def exclude_features_sparsify_Data_list(
    data_list: List[Data],
    target_tensor_dtype,
    exclude_feat_idx: Optional[List[int]] = None,
    sparsify_feats: bool = False,
    use_k_drop: bool = False
) -> List[Data]:
    """
    Given a list of (graphs stored in) PyG Data objects,
    this function (1) converts the target tensors to the
    desired dtype; (2) removes any desired features to 
    exclude; and (3) converts the node features (Data.x)
    to sparse tensors, if desired.

    Args:
        data_list: list of PyG Data objects.
        target_tensor_dtype: torch.dtype for target
            tensors (e.g. torch.float).
        exclude_feat_idx: optional list of indices for
            features to exclude from each Data object
            node feature tensor (Data.x).
        sparsify_feats: whether to convert node feature
            matrix tensors to torch sparse tensors.
        use_k_drop: whether to use k_drop methods (need
            to use k_drop.KDropData objects instead of
            Data objects).
    Returns:
        Processed list of PyG Data objects.
    """
    if exclude_feat_idx is not None:
        orig_num_feats = data_list[0].x.shape[1]
        incl_mask = torch.ones(orig_num_feats, dtype=torch.bool)
        incl_mask[exclude_feat_idx] = False
        xs = [
            g.x[:, incl_mask].to(target_tensor_dtype) \
            for g in data_list
        ]
    else: 
        xs = [
            g.x.to(target_tensor_dtype) \
            for g in data_list
        ]

    data_list = [
        Data(
            x=x.to_sparse() if sparsify_feats else x, 
            edge_index=g.edge_index, 
            y=g.y.to(target_tensor_dtype)
        ) \
        for x, g in zip(xs, data_list)
    ]
    return data_list


def get_attribute_global_col_mins_maxs_for_Data_list(
    data_list: List[Data], 
    attr_name: str = 'x'
) -> Tuple[torch.Tensor]:
    """
    Finds the global, by-column mins and maxes of a 
    matrix tensor feature/attribute shared by all 
    graphs in a list of pytorch-geometric Data objects.

    Args:
        data_list: list of pytorch-geometric Data
            objects with a shared tensor feature 
            attribute.
        attr_name: string key for the attribute;
            defaults to 'x', the node features
            tensor.
    Returns:
        2-tuple of mins and maxs tensors (each of which
        has length equal to the number of columns in the
        attribute tensor).
    """
    first_tensor = getattr(data_list[0], attr_name)
    min_vals = first_tensor.min(dim=0).values
    max_vals = first_tensor.max(dim=0).values

    for data in data_list[1:]:
        tensor = getattr(data, attr_name)
        min_vals = torch.minimum(min_vals, tensor.min(dim=0).values)
        max_vals = torch.maximum(max_vals, tensor.max(dim=0).values)

    return min_vals, max_vals


def min_max_scale_data_list(
    data_list: List[Data], 
    min_vals: torch.Tensor,
    max_vals: torch.Tensor,
    attr_name: str = 'x'
) -> None: # sets new attribute values in place
    """
    Uses mins and maxs (such as those found by 
    'get_attribute_global_col_mins_maxs_for_Data_list')
    to min-max scale a shared attribute of each Data
    object (in place) in a list of Data objects.

    Args:
        data_list: list of pytorch-geometric Data
            objects with a shared tensor feature 
            attribute.
        min_vals: tensor of minimum values for each
            column in the common attribute tensor
            across all Data objects in the list.
        max_vals:tensor of maximum values for each
            column in the common attribute tensor
            across all Data objects in the list.
        attr_name: string key for the attribute;
            defaults to 'x', the node features
            tensor.
    Returns:
        None; re-assigns min-max scaled attribute
        tensors in-place on the Data objects in the
        data_list.
    """
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0  # Prevent divide-by-zero

    for data in data_list:
        attr = getattr(data, attr_name)
        scaled_tensor = (attr - min_vals) / range_vals
        setattr(data, attr_name, scaled_tensor)



# def torch_sparse_identity(size: int) -> torch.Tensor:
#     """
#     Constructs an identity matrix tensor in sparse
#     COO form.

#     Args:
#         size: number of rows/columns in the (square)
#             identity matrix to be created.
#     Returns:
#         Sparse identity matrix tensor in COO format.
#     """
#     indices = torch.arange(size).unsqueeze(0).repeat(2, 1)
#     values = torch.ones(size)
#     return torch.sparse_coo_tensor(indices, values, (size, size))


# DEPRECATED: use 'get_P' from 'process_pyg_data.py' instead
# (THIS HAS COLUMN-NORMALIZATION, WHICH IS WRONG FOR Px multiplication)
# def get_Batch_P_sparse(
#     edge_index: torch.Tensor, 
#     edge_weight: Optional[torch.Tensor] = None,
#     n_nodes: int = None,
#     lazy: bool = True,
#     device: Optional[str] = None
# ) -> torch.Tensor:
#     r"""
#     Computes P, the lazy random walk diffusion 
#     operator on a graph defined as 
#     $$P = 0.5 (I - AD^{-1})$$,
#     where the graph is the disconnected batch
#     graph of a torch_geometric Batch object.

#     WARNING: leaving 'n_nodes' as None has the
#     'to_torch_coo_tensor' method infer the size
#     of A_sparse, etc. It may get it wrong in edge
#     cases; it is best to provide this value.

#     Args:
#         edge_index: edge_index (e.g., from a
#             pytorch_geometric Data or Batch object).
#         edge_weight: edge_weight (e.g., from a
#             pytorch_geometric Data or Batch object).
#         n_nodes: total number of nodes in batch 
#             'x' tensor (e.g., from a pytorch_geometric 
#             Batch object). 
#         lazy: whether to return lazy random walk operator
#             matrix.
#         device: string device key (e.g., 'cpu', 'cuda', 
#             'mps') for placing the output tensor; if
#             None, will check for cuda, else assign to cpu.
#     Returns:
#         Sparse P matrix tensor, of shape 
#         (N, N), where N = data.x.shape[0], 
#         the  total number of nodes across all
#         batched graphs. Note P_sparse is 'doubly
#         sparse': sparse off of block diagonals,
#         and each block is itself a sparse operator
#         P_i for each graph x_i.
#     """
#     from torch_geometric.utils import to_torch_coo_tensor
#     if device is None:
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # should we use 'torch.sparse_coo_tensor' here instead?
#     A_sparse = to_torch_coo_tensor(
#         edge_index, 
#         edge_weight,
#         (n_nodes, n_nodes)
#     ).to(device)
#     D = A_sparse.sum(dim=1).to_dense()
#     # as of Oct 2024, 'torch.sparse.spdiags' doesn't work on cuda 12.4,
#     # -> use function with cpu tensors, then move resulting sparse
#     # tensor to device
#     D = D.squeeze().to('cpu')
#     # for nodes with degree 0, prevent division by 0 error
#     D_inv = torch.where(D > 0, (1. / D), 0.)
#     D_inv = torch.sparse.spdiags(
#         diagonals=D_inv, 
#         offsets=torch.zeros(1).long().to('cpu'),
#         shape=(len(D), len(D))
#     ).to(device)
#     I = torch_sparse_identity(len(D)).to(device)
#     P_sparse = (I + torch.sparse.mm(A_sparse, D_inv)) # .to(device)
#     if lazy:
#         P_sparse = 0.5 * P_sparse
#     # P_sparse = P_sparse.coalesce()
#     return P_sparse


def get_Batch_Wjxs(
    x: Batch,
    P_sparse: torch.Tensor,
    scales_type: Literal['dyadic', 'custom'] = 'dyadic',
    diffusion_scales: Optional[torch.Tensor] = None,
    J: int = 5,
    include_lowpass: bool = True,
    filter_stack_dim: int = -1,
    rescale_filtrations: bool = False,
    # rescale_method: Literal['standardize', 'minmax'] = 'standardize',
    # vector_dim: Optional[int] = None
) -> torch.Tensor:
    r"""
    Computes diffusion wavelet filtrations on a disconnected 
    graph, using recursive sparse matrix multiplication of a 
    diffusion operator matrix P. This method skips computing 
    increasingly dense powers of P, which get denser with as 
    the power increases, by these steps:
    
    1. Compute $y_t = P^t x$ recursively via $y_t = P y_{t-1}$,
       (only using P, and not its powers, which grow denser).
    2. Subtract $y_{2^{j-1}} - y_{2^{j}}$ [dyadic scales]. 
        The result is $W_j x = (P^{2^{j-1}} - P^{2^j}) x$.
        (Thus, we never form the matrices P^t, t > 1, which get 
        denser with as the power increases.)

    Note that if 'diffusion_scales' is not None, using its custom
    scales will override the 'scales_type' parameter.
    
    Args:
        x: stacked node-by-channel (N, c) data matrix for a 
            disconnected  batch graph of a pytorch geometric 
            Batch object. 
        P_sparse: sparse diffusion operator matrix 
            for disconnected batch graph of a pytorch
            geometric Batch object (should be row-normalized
            for multiplication with x on the right).
        scales_type: 'dyadic' or 'custom' or None for fixed P^1.
        diffusion_scales: tensor of shape (n_scale_split_ts)
            or (n_channels, n_scale_split_ts) for calculating custom
            wavelet scales, containing the indices of ts 0...max($t$).
            Scales are constructed uniquely for each channel of x from
            $t$s with adjacent indices in rows of this tensor. If None,
            this function defaults to dyadic scales.
        J: max wavelet filter order, for dyadic scales. For example,
            $J = 4$ will give $T = 2^4 = 16$ max diffusion step.
        include_lowpass: whether to include the 
            'lowpass' filtration, $P^{2^J} x$.
        filter_stack_dim: new dimension in which to 
           stack Wjx (filtration) tensors.
        rescale_filtrations: whether to rescale the filtrations tensor. 
            Note that wavelet versus low-pass filtrations can have
            very different scales, so it can be useful to rescale. Note
            that this uses per-batch statistics, not global statistics, 
            or learnable as with nn.BatchNorm[n]d modules.
        rescale_method: [DEPRECATED]'standardize' (mean 0 std 1) or 'minmax' (onto
            interval [-1,1]).
        vector_dim: [DEPRECATED] if provided and the input corresponds to flattened
            vector features (shape (N*d, ...)), rescaling is applied
            independently for each coordinate dimension so that each
            component (e.g. x, y, z) keeps its own statistics.
    Returns:
        Dense tensor of shape (batch_total_nodes, n_channels,
        n_filtrations) = 'Ncj'.
    """

    # --- Helper function for shared powers ---
    def _get_Wjxs_from_shared_powers(
        x: torch.Tensor,
        P_sparse: torch.Tensor,
        powers_to_save: torch.Tensor,
        range_upper_lim: int,
        device: str
    ) -> torch.Tensor:
        Ptxs = [x.to(device)]
        Ptx = x.to(device)
        
        # calc P^t x for t \in 1...T, saving only needed P^txs
        # print('P_sparse.shape', P_sparse.shape)
        # print('Ptx.shape', Ptx.shape)
        for j in range(1, powers_to_save[-1] + 1):
            try:
                Ptx = torch.sparse.mm(P_sparse, Ptx)
                if j in powers_to_save:
                    # print(f"j={j}")
                    # it's possible the same power is in a
                    # custom 'diffusion_scales' more than once
                    if diffusion_scales is not None:
                        j_ct = (powers_to_save == j).sum().item()
                        for _ in range(j_ct):
                            Ptxs.append(Ptx.to(device))
                    else:
                        Ptxs.append(Ptx.to(device))
            except Exception as e:
                print(f"j={j}")
                raise e

        # print('len(Ptxs):', len(Ptxs))
        Wjxs = [Ptxs[j - 1] - Ptxs[j] for j in range(1, range_upper_lim)] # J + 2)]
        if include_lowpass:
            Wjxs.append(Ptxs[-1])
        Wjxs = torch.stack(Wjxs, dim=filter_stack_dim).to(device)
        return Wjxs
    
    # --- Main function body ---
    device = x.device

    # first option: custom unique scales for each channel
    # print('get_Batch_Wjxs: diffusion_scales:', diffusion_scales)
    if (diffusion_scales is not None) \
    and (diffusion_scales != 'dyadic') \
    and (diffusion_scales.dim() == 2):
        Ptxs = [x.to(device)]
        Ptx = x.to(device)
        
        # calc P^t x for t \in 1...T, saving all powers of t
        custom_scales_max_t = int(2 ** J) # e.g. J = 5 -> 32
        for j in range(1, custom_scales_max_t + 1):
            Ptx = torch.sparse.mm(P_sparse, Ptx)
            Ptxs.append(Ptx.to_dense().to(device))

        # compute filtrations ('Wjxs')
        # note that filter (P^u - P^v)x = (P^u x) - (P^v x)
        # here indexes for (P^u x) and (P^v x) within 'Ptxs' for each
        # channel are adjacent entries in each channel's 't_is'
        Wjxs = torch.stack([
            torch.stack([
                # as of Nov 2024, bracket slicing doesn't work with sparse tensors
                # patch: entries of 'Ptxs' made dense above, when added to Ptxs
                Ptxs[t_is[t_i - 1]][:, c_i] - Ptxs[t_is[t_i]][:, c_i] \
                for t_i in range(1, len(t_is))
            ], dim=-1) \
            for c_i, t_is in enumerate(diffusion_scales)
        ], dim=1).to(device)
        
        '''
        Wjxs = [None] * diffusion_scales.shape[0]
        for c_i, t_is in enumerate(diffusion_scales):
            channel_Wjxs = [None] * (len(t_is) - 1)
            c_i_tensor = torch.tensor([c_i]).to(device)
            
            for t_i in range(1, len(t_is)):
                Pu = torch.index_select(Ptxs[t_is[t_i - 1]], 1, c_i_tensor)
                Pv = torch.index_select(Ptxs[t_is[t_i]], 1, c_i_tensor)
                channel_Wjxs[t_i - 1] = (Pu - Pv).squeeze()
                # print('channel_Wjxs.shape:', channel_Wjxs[t_i - 1].shape)
                
            channel_Wjxs = torch.stack(channel_Wjxs, dim=-1)
            Wjxs[c_i] = channel_Wjxs
        Wjxs = torch.stack(Wjxs, dim=1)
        print('Wjxs.shape:', Wjxs.shape)
        '''
        
        # lowpass = P^T x, for all channels
        if include_lowpass:
            # print('Ptxs[-1].shape:', Ptxs[-1].shape)
            Wjxs = torch.concatenate(
                (Wjxs, Ptxs[-1].unsqueeze(dim=-1).to(device)), 
                dim=-1
            )
        # Wjxs shape (N, n_channels, n_filters)
        # print('Wjxs.shape:', Wjxs.shape)

    # second option: one set of custom scales shared by all channels
    # (custom is priority if passed)
    elif (diffusion_scales is not None) \
    and (diffusion_scales != 'dyadic') \
    and (diffusion_scales.dim() == 1):
        powers_to_save = diffusion_scales
        range_upper_lim = diffusion_scales.shape[0]
        # print('shared_powers_to_save:', shared_powers_to_save)
        # print('range_upper_lim:', range_upper_lim)
        Wjxs = _get_Wjxs_from_shared_powers(
            x=x,
            P_sparse=P_sparse,
            powers_to_save=powers_to_save,
            range_upper_lim=range_upper_lim,
            device=device
        )

    # third option: dyadic scales shared by all channels
    elif (scales_type == 'dyadic'):
        powers_to_save = 2 ** torch.arange(J + 1)
        range_upper_lim = J + 2
        Wjxs = _get_Wjxs_from_shared_powers(
            x=x,
            P_sparse=P_sparse,
            powers_to_save=powers_to_save,
            range_upper_lim=range_upper_lim,
            device=device
        )

    # fourth option: fixed P^1
    elif (diffusion_scales is None) and (scales_type is None):
        Ptx = x.to(device)
        Ptx = torch.sparse.mm(P_sparse, Ptx)
        Wjxs = Ptx.unsqueeze(dim=-1).to(device)
    else:
        raise NotImplementedError(
            f"No method implemented for scales_type={scales_type}"
        )
        
    # -----------------------------------------------------
    # Vector features: rescale per coordinate + filter
    #   Wjxs shape (N*d, C, F) -> (N, d, C, F)
    #   Reduce dims = (0, 2)  (nodes, channels)
    # Scalars (or unknown d): reduce dims = (0, 1)
    # -----------------------------------------------------

    if rescale_filtrations:
        raise NotImplementedError(
            "Rescaling filtrations within 'get_Batch_Wjxs' has been deprecated."
        )
        '''
        # --- Rescaling helper functions ---
        def _standardize(
            t: torch.Tensor, 
            dims: Tuple[int] = (0, 1)
        ) -> torch.Tensor:
            """
            Standardize tensor *per filter* (dim=-1) across the 
            given dims.
            """
            mean_val = t.mean(dim=dims, keepdim=True)
            std_val = t.std(dim=dims, keepdim=True)
            std_val = torch.where(std_val == 0, torch.ones_like(std_val), std_val)
            return (t - mean_val) / std_val

        def _minmax_scale(
            t: torch.Tensor, 
            dims: Tuple[int] = (0, 1),
            center_at_zero: bool = True
        ) -> torch.Tensor:
            """
            Min-max scale tensor *per filter* (dim=-1) across the 
            given dims. Center at zero if 'center_at_zero' is True.
            """
            min_val = t.min(dim=dims, keepdim=True).values
            max_val = t.max(dim=dims, keepdim=True).values
            range_val = max_val - min_val
            range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
            t = (t - min_val) / range_val  # -> [0,1]
            if center_at_zero:
                t = 2. * t - 1.            # -> [-1,1]
            return t

        # Vector case: rescale per coordinates and filters
        if (vector_dim is not None) and (Wjxs.shape[0] % vector_dim == 0):
            print("WARNING: rescaling vector features per-coordinate is probably not what you want...")
            N = Wjxs.shape[0] // vector_dim
            C, F = Wjxs.shape[1], Wjxs.shape[2]
            Wjxs_reshaped = Wjxs.view(N, vector_dim, C, F)
            reduce_dims = (0, 2)  # nodes, channels

            if rescale_method == 'standardize':
                Wjxs_reshaped = _standardize(Wjxs_reshaped, dims=reduce_dims)
            elif rescale_method == 'minmax':
                Wjxs_reshaped = _minmax_scale(Wjxs_reshaped, dims=reduce_dims)
            else:
                raise ValueError(
                    f"Unknown rescale option '{rescale_method}'. Choose 'standardize' or 'minmax'."
                )

            Wjxs = Wjxs_reshaped.view(N * vector_dim, C, F)

        else:
            # Scalar case: reduce over nodes and channels per filter
            if rescale_method == 'standardize':
                Wjxs = _standardize(Wjxs, dims=(0, 1))
            elif rescale_method == 'minmax':
                Wjxs = _minmax_scale(Wjxs, dims=(0, 1))
            else:
                raise ValueError(
                    f"Unknown rescale option '{rescale_method}'. Choose 'standardize' or 'minmax'."
                )
    '''

    return Wjxs



def channel_pool(
    channel_pool_key: str,
    x: torch.Tensor,
    num_graphs: int,
    batch_index: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Pools a feature tensor's channels (within
    channels, across nodes), using 'sum', 'max',
    'mean', or 'max+mean' methods.

    Args:
        channel_pool_key: string key for the type
            of channel pooling to apply: 'sum', 'max',
            'mean', or 'max+mean'.
        x: tensor of batched hidden node features, of 
            shape (total num. nodes, num. channels).
        num_graphs: number of graphs in the batch.
        batch_index: the Batch.batch attribute (the
            tensor with each graph's index at identifying
            which rows of the batched feature matrix belong
            to the ith graph).
    Returns:
        Tensor of the hidden features pooled within channels.
    """
    if channel_pool_key not in (
        'sum', 
        'max', 
        'mean', 
        'max+mean', 
        'mean+max'
    ):
        raise NotImplementedError(
            f"'{channel_pool_key}' channel pooling not yet implemented."
        )
        
    x_i_chan_pools_max = [None] * num_graphs
    x_i_chan_pools_mean = [None] * num_graphs
    for i in range(num_graphs):
        if num_graphs > 1:
            # subset out ith graph
            x_i_mask = (batch_index == i)
            x_i = x[x_i_mask]
        else:
            x_i = x
        if 'sum' in channel_pool_key:
            x_i_chan_pools_mean[i] = torch.sum(x_i, dim=0)
        if 'max' in channel_pool_key:
            x_i_chan_pools_max[i] = torch.max(x_i, dim=0).values
        if 'mean' in channel_pool_key:
            x_i_chan_pools_mean[i] = torch.mean(x_i, dim=0)

    if ('max' in channel_pool_key) \
    and ('mean' not in channel_pool_key):
        x = torch.stack(x_i_chan_pools_max)
        # x shape: (n_graphs, [final_]n_channels_3)
        
    if ('mean' in channel_pool_key) \
    and ('max' not in channel_pool_key):
        x = torch.stack(x_i_chan_pools_mean)
        # x shape: (n_graphs, [final_]n_channels_3)
        
    if ('mean' in channel_pool_key) \
    and ('max' in channel_pool_key):
        maxs = torch.stack(x_i_chan_pools_max)
        means = torch.stack(x_i_chan_pools_mean)
        x = torch.stack((maxs, means), dim=1)
        # x shape: (n_graphs, 2, [final_]n_channels_3)
    return x


def moments_channel_pool(
    x: torch.Tensor,
    batch_index: Optional[torch.Tensor],
    num_graphs: int,
    channel_pool_moments: Tuple[int] = (1, 2, 3, 4),
    rescale_moments: bool = False
) -> torch.Tensor:
    """
    Pools a feature tensor's channels into moments (across
    nodes).

    Args:
        x: feature tensor of shape (N, c), where N is the 
            total number of nodes in the batched graphs,
            and c is the number of (hidden) features/channels.
        batch_index: 1-d batch index tensor of length N
            that identifies individual graph's node indices 
            in a py-g Batch object, which collates graphs into 
            one large disconnected graph. E.g.,
            [0, ..., 0, 1, ..., 1, ..., n-1, ..., n-1]
        num_graphs: number of individual graphs in the Batch.
        channel_pool_moments: tuple of moments to return, e.g.
            (1, 2, 3, 4) returns the 1st through 4th moments.
        rescale_moments: if True, rescales moments
            (across graphs, within moment-channels)
            onto interval [-1, 1], so that all new moment
            pooled features values are on the same
            scale. Note that this centers all moment 
            features at 0, losing any cross-channel differences 
            in their distribution centers. Hence, use with
            caution: the gain in numerical optimization
            might come at the cost of hindered learning.

    Returns:
        Tensor of containing moments of each channel
        in the original input x, shape (b, Q, c), where
        b is the number of graphs in the batch, Q is 
        the number of moments, and c is the number of
        channels.
    """
    # pool individual graph's channels as moments
    # (across nodes)
    q_norms = [None] * num_graphs
    for i in range(num_graphs):

        if num_graphs > 1:
            # subset out ith graph
            x_i_mask = (batch_index == i)
            x_i = x[x_i_mask]
        else:
            x_i = x
    
        # compute q-norms of its columns
        # technically, moments as defined in MFCN are ||Wjx||_q^q,
        # so we should raise norms to their qs here, but it shouldn't matter
        # for learning
        x_i_q_norms = torch.stack([
            linalg.vector_norm(
                x=x_i, 
                ord=q, 
                dim=0 # 0 for norms of col vecs
            ) ** q \
            for q in channel_pool_moments
        ]) # x_i_q_norms shape: (Q, n_channels)
        q_norms[i] = x_i_q_norms

    # stack all graph's moments
    x = torch.stack(q_norms)
    # x shape = (n_graphs, Q, n_channels)

    if rescale_moments:
        # compute min and max along the first dimension (n_graphs), keeping dimensions
        min_vals = x.min(dim=0, keepdim=True).values  # shape: (1, Q, n_channels)
        max_vals = x.max(dim=0, keepdim=True).values  # shape: (1, Q, n_channels)
    
        # avoid division by zero in case max == min
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # to prevent division by zero
    
        # apply min-max scaling -> interval [0, 1]
        x = (x - min_vals) / range_vals

        # -> interval [-1, 1]
        x = 2. * x - 1.
    
    return x


def node_pool(
    x: torch.Tensor, 
    node_pooling_key: str, 
    node_pool_wts: Optional[torch.Tensor],
    node_pool_bias: Optional[torch.Tensor],
    pool_dim: int = 1
) -> torch.Tensor:
    """
    Applies various simple node pooling operations
    to a tensor of node features.

    Args:
        x: tensor of hidden node features.
        node_pooling_key: string key for node
            pooling type: 'mean', 'max', 'sum', 
            or 'linear' (for a simple linear layer).
        node_pool_wts: a torch.Parameter tensor of
            learnable node pooling linear layer weights.
        node_pool_bias: a torch.Parameter tensor of
            learnable node pooling linear layer bias.
        pool_dim: the tensor 'dim' (dimension index)
            at which to perform mean, sum, or max 
            node pooling.
    Returns:
        Tensor of node pooling values.
    """
    if 'mean' in node_pooling_key:
        x = torch.mean(x, dim=pool_dim)
    elif 'sum' in node_pooling_key:
        x = torch.sum(x, dim=pool_dim)
    elif 'max' in node_pooling_key:
        x = torch.max(x, dim=pool_dim).values
    elif 'linear' in node_pooling_key:
        # channels are linearly-combined within nodes,
        # x' = wx + b, using the same learned w and b
        # parameters for all nodes
        # (N, d) @ (d, 1) -> (N, 1)
        x = torch.matmul(x, node_pool_wts) + node_pool_bias
    else:
        raise NotImplementedError(
            f"'{node_pooling_key}' node pooling method not implemented."
            f"Did you mean 'mean', 'sum', 'max', or 'linear'?"
        )
    return x

