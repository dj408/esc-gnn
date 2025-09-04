"""
Functions for testing ESCGNN on a 2-d vector field.
"""
import numpy as np
import torch
from numpy.random import RandomState
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from typing import (
    Tuple,
    List,
    Dict,
    Optional,
    Callable,
    Iterable,
    Any
)


def vector_field(x, y):
    u = np.cos(2 * np.pi * (x + y))
    v = np.sin(2 * np.pi * (x - y))
    return u, v


def sample_points(n, random_state, domain=[-1, 1]):
    x_samples = random_state.uniform(domain[0], domain[1], n)
    y_samples = random_state.uniform(domain[0], domain[1], n)
    return x_samples, y_samples


def plot_vector_field(
    x, y, u, v, 
    title: str = "", 
    color: str = "b",
    alpha: float = 1.0
) -> None:    
    # plt.figure(figsize)
    plt.quiver(
        x, y, u, v, 
        color=color, 
        alpha=alpha,
        angles='xy'
    )
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid()
    # plt.show()


def plot_graph(xy, edge_index):
    # plot edges
    x, y = xy[:, 0], xy[:, 1]
    for i in range(edge_index.shape[1]):
        src, tgt = edge_index[:, i]
        x0, y0 = x[src], y[src]
        x1, y1 = x[tgt], y[tgt]
        plt.xlim(-1.05, 1.05)
        plt.ylim(-1.05, 1.05)
        plt.plot(
            [x0, x1], 
            [y0, y1], 
            '-',
            c='black',
            alpha=0.5
        )
    
    # plot nodes
    for x, y in xy:
        plt.scatter(x, y, c='black')
    
    # display graph plot
    # plt.show()
    
    

def rotation_2d(
    theta: float,
    points: Optional[np.ndarray | torch.Tensor] = None,
    matrix_constructor: str = 'tensor'
) -> np.ndarray | torch.Tensor:
    """
    Returns a 2D rotation matrix for a given angle theta,
    or an array/tensor of rotated points if `points is
    not None`. (Works with both PyTorch tensors and NumPy 
    arrays, but defaults to tensors.)

    Args:
        theta: rotation angle in radians.
        points: array or tensor of shape 2 x N, where
            N is the number of points.
        matrix_constructor: if no points are provided
            and only the R matrix is desired as a numpy
            array, set this parameter to 'arr'.
    Returns:
        Rhe 2x2 rotation matrix tensor if no points are 
        passed, otherwise, the rotated points, as a numpy 
        array or torch tensor of shape (Nx2).
    """
    if ((points is not None) and isinstance(points, np.ndarray)) \
    or ((points is None) and ('arr' in matrix_constructor.lower())):
            cos, sin = np.cos, np.sin
            matrix_constructor = np.array
    else: # otherwise, use tensors
        theta = torch.as_tensor(theta, dtype=points.dtype) \
            if points is not None \
            else torch.as_tensor(theta)
        cos, sin = torch.cos, torch.sin
        matrix_constructor = torch.tensor
    
    cos_t, sin_t = cos(theta), sin(theta)
    R = matrix_constructor(
        [[cos_t, -sin_t],
         [sin_t,  cos_t]]
    )
    if points is not None:
        return R @ points
    else: 
        return R


def construct_knn_graph(
    x: np.ndarray, 
    y: np.ndarray, 
    u: np.ndarray, 
    v: np.ndarray,
    k: int = 5,
    vector_feat_key: str = 'v'
) -> Data:
    """
    Constructs a k-NN graph based on x and y
    point coordinates with 2-d vector node
    features (u and v), and contains in a
    pytorch-geometric Data object.
    """
    points = np.vstack((x, y)).T
    knn_graph_matrix = kneighbors_graph(
        points,
        k,
        mode='connectivity',
        include_self=False
    )
    edge_index = torch.tensor(
        np.array(knn_graph_matrix.nonzero()), 
        dtype=torch.long
    )
    # 'v' is the vector features matrix; should have shape (n, d),
    # where n is the number of nodes and d is the dimension of the node vectors
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u)
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v)
    v = torch.stack((u, v),dim=-1)
    data = Data(
        x=torch.tensor(points, dtype=torch.float),
        edge_index=edge_index,
    )
    data[vector_feat_key] = v
    return data

