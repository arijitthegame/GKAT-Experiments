# pylint: skip-file

import numpy as np 
from tqdm import trange
import torch 

from lanczos import lanczos
from conjugate_gradient import cg_batch
from sdd_solver import *

#TODO: ADD SPARSE FROM HAN
def compute_lanczos_matrix_exp(A, v, k, use_reorthogonalization=False, return_exp=False):
    """
    Compute the action of matrix exponential on a vector v using the Lanczos algorithm.
    Can also optionally return the approximate exponential matrix too
    This is figure 4 in https://arxiv.org/abs/1111.1491
    v is assumed to be of shape B x N

    #TODO: ADD NECESSARY CHECKS FOR SHAPES
    """

    if len(v.shape) == 1 :
      v = v.unsqueeze(0)

# compute Q, T via Lanczos, T is the tridiagonal matrix with shape k x k and Q is of shape n x k
    T, Q = lanczos(A, num_eig_vec=k, mask=None, use_reorthogonalization=use_reorthogonalization)

    D, P = torch.linalg.eigh(T) 
    exp_T = torch.bmm(torch.bmm(P, torch.diag_embed(torch.exp(D))), P.transpose(1, 2))

    #compute the action

    exp_A = torch.bmm(torch.bmm(Q, exp_T),  Q.transpose(1, 2))

    w = torch.einsum('ijk, ik -> ij', exp_A, v)

    if return_exp is False:
      return w
    else: 
      return w, exp_A


def compute_exprational_matrix_exp(A, v, k, return_exp=False, method_type='cg', tolerance=1e-6, eps=1e-15):
    '''
    This is the algorithm as described in figure 5 of https://arxiv.org/abs/1111.1491
    SDD solver in the paper is not implemented however we can use CG or Gauss-Siedel or Jacobi methods instead
    Args: A matrix A of shape b x n x n, 
        v vector of shape B x n, 
        k number of eigenvectors to compute, 
        return_exp=True if you want to return the approximate exponential matrix
        method_type = 'cg' or 'jacobi' or 'gauss_seidel'

    A vector u such that ||exp(-A)v - u|| ≤ ε
    '''
    #TODO: CHECK SHAPES

    A = [None] * (k+1)
    V = [None] * (k+1)
    for i in trange(k):
      if method_type == 'cg':
        V[0] = v
        w, = cg_batch(A, v, r_tol=tolerance)
        for j in range(i):
          A[j][i] = torch.einsum('ij, kj -> ik', v, w)

        #orthogonalize
        w = w - torch.einsum('ij, kj -> ik', A[j][i], v)
        A[i+1][i] = torch.norm(w, dim=1) + eps
        V[i+1] = w / A[i+1][i]

      else:
        raise NotImplementedError("Other methods are not yet implemented")

    T_hat = .5 * (A + A.transpose(1,2))
    # B = exp (k (I − T^{-1}) exactly  
    # WHAT IS THE EASISET WAY TO INVERT T_hat???

    #TODO: FINISH THIS

    return NOne


      





if __name__ == "__main__":
    A =  A = torch.randn(3, 6400, 6400)
    A = A + A.transpose(1,2)
    v = torch.nn.functional.normalize(torch.rand(3, 6400), dim =1)
    w = compute_lanczos_matrix_exp(A, v, k=64)