import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as solver
from damped_jacobi import damped_jacobi
from restriction_prolongation import restriction_2D_operator, prolongation_2D_operator


def multigrid_alg(A, u, b, level, gridSize, smoothing_iterations, omega):
    if level == 0 or b.shape[0] <= 4:
        return sp.linalg.spsolve(A, b)

    # pre-relaxation
    u = damped_jacobi(A, b, u, smoothing_iterations, omega)

    residual = b - A @ u
    R = restriction_2D_operator(gridSize)
    P = prolongation_2D_operator(gridSize)
    A_2h = R @ A @ P
    b_2h = R @ residual
    coarse_gridSize = (gridSize + 1) // 2
    u_2h = np.zeros((coarse_gridSize-1)**2)
    e_2h = multigrid_alg(A_2h, u_2h, b_2h, level-1, coarse_gridSize, smoothing_iterations, omega)
    e_h = P @ e_2h
    u += e_h

    # post-relaxations
    u = damped_jacobi(A, b, u, smoothing_iterations, omega)
    return u