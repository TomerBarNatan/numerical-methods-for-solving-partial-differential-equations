import numpy as np
import scipy.sparse as sp

def laplacian2D(gridSize, h):
    L1D = sp.diags([np.ones(gridSize) * -2 / (h ** 2), np.ones(gridSize - 1) / (h ** 2), np.ones(gridSize - 1) / (h ** 2)], [0, 1, -1])
    L2D = sp.kron(np.eye(gridSize), L1D) + sp.kron(L1D, np.eye(gridSize))
    return L2D