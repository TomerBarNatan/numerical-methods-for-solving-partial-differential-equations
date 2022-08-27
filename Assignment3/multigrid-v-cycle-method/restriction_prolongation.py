import numpy as np
import scipy.sparse as sp

def restriction_1D_operator(gridSize):
    I = np.zeros(((gridSize - 1) // 2, gridSize - 1))
    for i in range(gridSize // 2 - 1):
        I[i][2 * i:2 * i + 3] = [1, 2, 1]
    I /= 4
    return I

def restriction_2D_operator(gridSize):
    I = restriction_1D_operator(gridSize)
    I = sp.kron(I,I)
    return I

def prolongation_2D_operator(gridSize):
    I = 2*restriction_1D_operator(gridSize).transpose()
    I = sp.kron(I,I)
    return I