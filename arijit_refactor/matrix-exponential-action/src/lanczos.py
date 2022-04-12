# pylint: skip-file

import torch
from torch.nn import functional as F

'''
Code adopted from lanczosnet https://arxiv.org/abs/1901.01484
'''

EPS = 1e-6

def lanczos(A, num_eig_vec, mask=None, use_reorthogonalization=False):
    """ Lanczos for symmetric matrix A
    
      Args:
        A: float tensor, shape B X N X N
        mask: float tensor, shape B X N
        num_eig_vec = K
      Returns:
      T: shape B X K X K, tridiagonal matrix
      Q: shape B X N X K, orthonormal matrix
      
    """
    batch_size = A.shape[0]
    num_node = A.shape[1]
    lanczos_iter = min(num_node, num_eig_vec)

    # initialization
    alpha = [None] * (lanczos_iter + 1)
    beta = [None] * (lanczos_iter + 1)
    Q = [None] * (lanczos_iter + 2)

    beta[0] = torch.zeros(batch_size, 1, 1).to(A.device)
    Q[0] = torch.zeros(batch_size, num_node, 1).to(A.device)
    Q[1] = torch.randn(batch_size, num_node, 1).to(A.device)

    if mask is not None:
        mask = mask.unsqueeze(dim=2).float()
        Q[1] = Q[1] * mask

    Q[1] = Q[1] / torch.norm(Q[1], 2, dim=1, keepdim=True)

    # Lanczos loop
    lb = 1.0e-4
    valid_mask = []
    for ii in range(1, lanczos_iter + 1):
      z = torch.bmm(A, Q[ii])  # shape B X N X 1
      alpha[ii] = torch.sum(Q[ii] * z, dim=1, keepdim=True)  # shape B X 1 X 1
      z = z - alpha[ii] * Q[ii] - beta[ii - 1] * Q[ii - 1]  # shape B X N X 1

      if use_reorthogonalization and ii > 1:
        # N.B.: Gram Schmidt does not bring significant difference of performance
        def _gram_schmidt(xx, tt):
          # xx shape B X N X 1
          for jj in range(1, tt):
            xx = xx - torch.sum(
                xx * Q[jj], dim=1, keepdim=True) / (
                    torch.sum(Q[jj] * Q[jj], dim=1, keepdim=True) + EPS) * Q[jj]
          return xx

        # do Gram Schmidt process twice
        for _ in range(2):
          z = _gram_schmidt(z, ii)

      beta[ii] = torch.norm(z, p=2, dim=1, keepdim=True)  # shape B X 1 X 1

      # N.B.: once lanczos fails at ii-th iteration, all following iterations
      # are doomed to fail
      tmp_valid_mask = (beta[ii] >= lb).float()  # shape
      if ii == 1:
        valid_mask += [tmp_valid_mask]
      else:
        valid_mask += [valid_mask[-1] * tmp_valid_mask]

      # early stop
      Q[ii + 1] = (z * valid_mask[-1]) / (beta[ii] + EPS)

    # get alpha & beta
    alpha = torch.cat(alpha[1:], dim=1).squeeze(dim=2)  # shape B X T
    beta = torch.cat(beta[1:-1], dim=1).squeeze(dim=2)  # shape B X (T-1)

    valid_mask = torch.cat(valid_mask, dim=1).squeeze(dim=2)  # shape B X T
    idx_mask = torch.sum(valid_mask, dim=1).long()
    if mask is not None:
      idx_mask = torch.min(idx_mask, torch.sum(mask, dim=1).squeeze().long())

    for ii in range(batch_size):
      if idx_mask[ii] < valid_mask.shape[1]:
        valid_mask[ii, idx_mask[ii]:] = 0.0

    # remove spurious columns
    alpha = alpha * valid_mask
    beta = beta * valid_mask[:, :-1]

    T = []
    for ii in range(batch_size):
      T += [
          torch.diag(alpha[ii]) + torch.diag(beta[ii], diagonal=1) + torch.diag(
              beta[ii], diagonal=-1)
      ]

    T = torch.stack(T, dim=0)  # shape B X T X T
    Q = torch.cat(Q[1:-1], dim=2)  # shape B X N X T
    Q_mask = valid_mask.unsqueeze(dim=1).repeat(1, Q.shape[1], 1)

    # remove spurious rows
    for ii in range(batch_size):
      if idx_mask[ii] < Q_mask.shape[1]:
        Q_mask[ii, idx_mask[ii]:, :] = 0.0

    Q = Q * Q_mask

    # pad 0 when necessary
    if lanczos_iter < num_eig_vec:
      pad = (0, num_eig_vec - lanczos_iter, 0,
             num_eig_vec - lanczos_iter)
      T = F.pad(T, pad)
      pad = (0, num_eig_vec - lanczos_iter)
      Q = F.pad(Q, pad)

    return T, Q

