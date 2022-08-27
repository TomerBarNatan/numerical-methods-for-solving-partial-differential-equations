import numpy as np
from laplacian2d import laplacian2D
from multigrid import multigrid_alg
from boundary_conditions import add_BC

if __name__ == '__main__':
    omega = 0.8
    for h in [0.05, 0.01]:
        gridSize = int(2 / h)
        grid1D = np.linspace(-1 + h, 1 - h, gridSize - 1)
        gridX, gridY = np.meshgrid(grid1D, grid1D)
        bc = add_BC(gridX, gridY)
        b = bc.flatten() / h ** 2
        lap2D = laplacian2D(gridSize - 1, h)
        u = np.zeros(b.shape)
        exact_sol = lambda x, y: (2 * (1 + y)) / ((3 + x) ** 2 + (1 + y) ** 2)
        actual_function = exact_sol(gridX, gridY)
        actual_function = actual_function.flatten()
        residual_vector = []
        error_vector = []
        for i in range(15):
            u = multigrid_alg(-lap2D, u, b, 3, gridSize, 1, omega)
            residual_vector.append(np.linalg.norm((-lap2D)@u - b))
            error_vector.append(np.max(abs(u-actual_function)))
