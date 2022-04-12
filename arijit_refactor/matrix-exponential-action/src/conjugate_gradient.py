import torch
import time
from einops import rearrange, repeat

'''
Code adapted from https://github.com/sbarratt/torch_cg with a bunch of my modifications. If bugs are found, please contact @arijitthegame
'''

def cg_batch(A_bmm, B, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
    """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.
    This function solves a batch of matrix linear systems of the form
        A_i X_i = B_i,  i=1,...,K,
    where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
    and X_i is the n x m matrix representing the solution for the ith system.
    Args:
        A_bmm: a K x n x m matrix.
        B: A K x n x m matrix representing the right hand sides.
        M_bmm: (optional) the preconditioning
            matrices M of shape K x n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
    K, n, m = B.shape

    if M_bmm is None:
        M_bmm = repeat(torch.eye(n), ' row col -> 1 row col')
    if X0 is None:
        X0 = torch.bmm(M_bmm, B)
    if maxiter is None:
        maxiter = 5 * n

    assert B.shape == (K, n, m)
    assert X0.shape == (K, n, m)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    X_k = X0
    R_k = B - torch.bmm(A_bmm,X_k)
    Z_k = torch.bmm(M_bmm, R_k)

    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    B_norm = torch.norm(B, dim=1)
    stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

    if verbose:
        print("%03s | %010s %06s" % ("it", "dist", "it/s"))

    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        start_iter = time.perf_counter()
        Z_k = torch.bmm(M_bmm, R_k)

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum(1)
            denominator[denominator == 0] = 1e-8
            beta = (R_k1 * Z_k1).sum(1) / denominator
            P_k = Z_k1 + beta.unsqueeze(1) * P_k1

        denominator = (P_k * torch.bmm(A_bmm, P_k)).sum(1)
        denominator[denominator == 0] = 1e-8
        alpha = (R_k1 * Z_k1).sum(1) / denominator
        X_k = X_k1 + alpha.unsqueeze(1) * P_k
        R_k = R_k1 - alpha.unsqueeze(1) * torch.bmm(A_bmm, P_k)
        end_iter = time.perf_counter()

        residual_norm = torch.norm(torch.bmm(A_bmm, X_k) - B, dim=1)

        if verbose:
            print("%03d | %8.4e %4.2f" %
                  (k, torch.max(residual_norm-stopping_matrix),
                    1. / (end_iter - start_iter)))

        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break

    end = time.perf_counter()

    if verbose:
        if optimal:
            print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                  (k, (end - start) * 1000))
        else:
            print("Terminated in %d steps (optimal). Took %.3f ms." %
                  (k, (end - start) * 1000))


    info = {
        "niter": k,
        "optimal": optimal
    }

    return X_k, info


class CG(torch.autograd.Function):
    #Throws error if A_bmm is set grad to be True 
        
    @staticmethod
    def forward(ctx, A_bmm, B, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
        
        ctx.A_bmm = A_bmm
        ctx.M_bmm = M_bmm
        ctx.rtol = rtol
        ctx.atol = atol
        ctx.maxiter = maxiter
        ctx.verbose = verbose
        
        X, _ = cg_batch(A_bmm, B, M_bmm, X0=X0, rtol=rtol,
                     atol=atol, maxiter=maxiter, verbose=verbose)
        ctx.save_for_backward(B, X)

        return X

    @staticmethod
    def backward(ctx, dX):

        
        B, X, = ctx.saved_tensors
        dB, _ = cg_batch(ctx.A_bmm, dX, ctx.M_bmm, rtol=ctx.rtol,
                      atol=ctx.atol, maxiter=ctx.maxiter, verbose=ctx.verbose)
        return dB, None
