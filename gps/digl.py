import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import to_dense_adj

def gdc(edge_index, alpha: float, eps: float):
    A = to_dense_adj(edge_index).squeeze()
    N = A.shape[0]

    # Self-loops
    A_loop = torch.eye(N).to(A.device) + A

    # Symmetric transition matrix
    D_loop_vec = torch.squeeze(torch.sum(A_loop,dim=1))
    D_loop_vec_invsqrt = 1 / torch.sqrt(D_loop_vec)
    D_loop_invsqrt = torch.diag(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt

    # PPR-based diffusion
    S = alpha * torch.linalg.inv(torch.eye(N).to(A.device) - (1 - alpha) * T_sym)

    # Sparsify using threshold epsilon
    S_tilde = (S >= eps)
    
    return S_tilde.nonzero().t().contiguous()



