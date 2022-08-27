import numpy as np

def f_x_y(x, y):
    return 10 * (1 + y) / (((3 + x) ** 2) + ((1 + y) ** 2))


def boundaryR(y):  # Right BC
    return 2 * (1 + y) / (4 + (1 + y) ** 2)


def boundaryL(y):  # Left BC
    return 2 * (1 + y) / (16 + (1 + y) ** 2)


def boundaryB(x):  # Bottom BC
    return 0

def boundaryU(x):  # Top BC
    return 4 / (((3 + x) ** 2) + 4)

def actualFunction(x, y):
    return 2 * (1 + y) / ((3 + x) ** 2 + (1 + y) ** 2)

def add_BC(x_mat, y_mat):
    BC = np.zeros(x_mat.shape)
    BC[:, 0] += boundaryR(y_mat[:, 0])
    BC[:, -1] += boundaryL(y_mat[:, -1])
    BC[0, :] += boundaryB(y_mat)
    BC[-1, :] += boundaryU(x_mat[-1, :])
    return BC