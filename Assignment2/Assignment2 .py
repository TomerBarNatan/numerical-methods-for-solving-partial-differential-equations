import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.pyplot import legend, semilogy
import scipy.sparse as sp
import scipy.sparse.linalg as solver

def f_x_y(x, y):
    return 10 * (1 + y) / (((3 + x) ** 2) + ((1 + y) ** 2))

def boundary1(y): #Right BC
    return 2 * (1 + y) / (4 + (1 + y) ** 2)


def boundary2(y): #Left BC
    return 2 * (1 + y) / (16 + (1 + y) ** 2)


def boundary3(x): #Bottom BC
    return 0


def boundary4(x): #Top BC
    return 4 / (((3 + x) ** 2) + 4)

def actualFunction(x, y):
    return 2 * (1 + y) / ((3 + x) ** 2 + (1 + y) ** 2)

def Poisson2D(h_y, h_x):
    ##Defining and flattering the grid
    gridSizeY = int(2 / h_y)
    gridSizeX = int(2 / h_x)
    h = h_x #Assuming that h_x=h_y at this section
    gridUX = np.linspace(-1 + h, 1-h, gridSizeX-1)
    gridUY = np.linspace(-1 + h, 1-h, gridSizeY-1)
    gridX, gridY = np.meshgrid(gridUX, gridUY)
    Xu = gridX.flatten()
    Yu = gridY.flatten()

    ##Adding BC to the solution vector
    b = np.zeros(gridX.shape)
    b[:,0] +=boundary1(gridY[:,0])
    b[:,-1] += boundary2(gridY[:,-1])
    b[0,:] += boundary3(gridX[0,:])
    b[-1, :] += boundary4(gridX[-1,:])
    b = b.flatten()
    
    ##Adding the given function to the solution vector
    for i in range(b.shape[0]):
        b[i] += f_x_y(Xu[i], Yu[i])*(h**2)
    
    ##Define the 1D-Laplacian
    main_diag = (-2)* np.ones(gridUX.shape[0])
    sub_diag = np.ones(gridUX.shape[0] - 1)
    lap1D = sp.coo_matrix(np.diag(sub_diag, k=-1) + np.diag(main_diag) + np.diag(sub_diag, k=1))
    
    ##Define the 2D-Laplacian
    Id_mat = sp.identity(gridUX.shape[0])
    lap2D = sp.kron(Id_mat, lap1D) + sp.kron(lap1D, Id_mat)
    lap2D = -lap2D +sp.identity(lap2D.shape[0])*5*(h**2)
    
    ##Solving the linear system and calculating the error
    approx_solution = solver.spsolve(lap2D, b)
    correct_solution = np.array([actualFunction(i, j) for i, j in zip(Xu, Yu)])
    error_vector = np.array(abs(correct_solution - approx_solution))
    relative_err = np.divide(error_vector, correct_solution, where=correct_solution != 0)
    relative_max_error = np.max(relative_err)
    return relative_max_error



errors = []
for h in [0.2,0.1, 0.05]:
    errors.append(Poisson2D(h, h))

plt.plot([0.2,0.1, 0.05], errors)
plt.xlabel('h_x=h_y values')
plt.ylabel('max relative error')
plt.title("Max relative error for discrete approcimation")




def Poisson2DJacobi(h_x, h_t):
    gridSizeY = int(2 / h_t)
    gridSizeX = int(2 / h_x)
    h = h_x
    gridUX = np.linspace(-1 + h, 1-h, gridSizeX-1)
    gridUY = np.linspace(-1 + h, 1-h, gridSizeY-1)
    gridX, gridY = np.meshgrid(gridUX, gridUY)
    Xu = gridX.flatten()
    Yu = gridY.flatten()
    b = np.zeros(gridX.shape)
    b[:,0] +=boundary1(gridY[:,0])
    b[:,-1] += boundary2(gridY[:,-1])
    b[0,:] += boundary3(gridX[0,:])
    b[-1, :] += boundary4(gridX[-1,:])
    b = b.flatten()
    for i in range(b.shape[0]):
        b[i] += f_x_y(Xu[i], Yu[i])*(h**2)
    correct_solution = np.array([actualFunction(i, j) for i, j in zip(Xu, Yu)])
    main_diag = (-2)* np.ones(gridUX.shape[0])
    sub_diag = np.ones(gridUX.shape[0] - 1)
    lap1D = sp.coo_matrix(np.diag(sub_diag, k=-1) + np.diag(main_diag) + np.diag(sub_diag, k=1))
    Id_mat = sp.identity(gridUX.shape[0])
    lap2D = sp.kron(Id_mat, lap1D) + sp.kron(lap1D, Id_mat)
    lap2D = -lap2D +sp.identity(lap2D.shape[0])*5*(h**2)
    JacobiIterations(lap2D, b, correct_solution)



def JacobiIterations(lap2D, b, solution):
    residual_vector = []
    error_vector = []
    lap2D = np.array(sp.coo_matrix.todense(lap2D)) 
    M = np.diag(lap2D)
    N = lap2D - np.diagflat(M)
    x = np.zeros(lap2D.shape[0])
    for i in range(100):
        x= (b-np.dot(N,x))/M
        residual_vector.append(np.linalg.norm(b - np.dot(lap2D,x)))
        error_vector.append(np.linalg.norm(solution - x))

        
    plt.semilogy([i for i in range(100)], residual_vector)
    plt.xlabel('Iteration number')
    plt.ylabel('Residual vector in l2 norm')
    plt.title("Residual norm in semi-logarithmic scale")
    plt.show()
    
    
    convergence_factor = [error_vector[i+1]/error_vector[i] for i in range(99)]
    plt.plot([i for i in range(99)], convergence_factor)
    plt.xlabel('Iteration number')
    plt.ylabel('Convergence factor value')
    plt.title("Convergence factor for Jacobi method")
    plt.show()
    
    
    
    
    residual_drop_factor = [residual_vector[i+1]/residual_vector[i] for i in range(99)]
    plt.plot([i for i in range(99)], residual_drop_factor)
    plt.xlabel('Iteration number')
    plt.ylabel('Residual drop factor value')
    plt.title("Residual drop factor for Gauss-Seidel method")
    plt.show()

Poisson2DJacobi(0.03,0.03)


def Poisson2DGaussSeidel(h_x, h_t):
    gridSizeY = int(2 / h_t)
    gridSizeX = int(2 / h_x)
    h = h_x
    gridUX = np.linspace(-1 + h, 1-h, gridSizeX-1)
    gridUY = np.linspace(-1 + h, 1-h, gridSizeY-1)
    gridX, gridY = np.meshgrid(gridUX, gridUY)
    Xu = gridX.flatten()
    Yu = gridY.flatten()
    b = np.zeros(gridX.shape)
    b[:,0] +=boundary1(gridY[:,0])
    b[:,-1] += boundary2(gridY[:,-1])
    b[0,:] += boundary3(gridX[0,:])
    b[-1, :] += boundary4(gridX[-1,:])
    b = b.flatten()
    for i in range(b.shape[0]):
        b[i] += f_x_y(Xu[i], Yu[i])*(h**2)
    correct_solution = np.array([actualFunction(i, j) for i, j in zip(Xu, Yu)])
    main_diag = (-2)* np.ones(gridUX.shape[0])
    sub_diag = np.ones(gridUX.shape[0] - 1)
    lap1D = sp.coo_matrix(np.diag(sub_diag, k=-1) + np.diag(main_diag) + np.diag(sub_diag, k=1))
    Id_mat = sp.identity(gridUX.shape[0])
    lap2D = sp.kron(Id_mat, lap1D) + sp.kron(lap1D, Id_mat)
    lap2D = -lap2D +sp.identity(lap2D.shape[0])*5*(h**2)
    GaussSeidelIterations(lap2D, b, correct_solution)


def GaussSeidelIterations(lap2D, b, solution):
    residual_vector = []
    error_vector = []
    lap2D = np.array(sp.coo_matrix.todense(lap2D)) 
    M = np.tril(lap2D)
    N = lap2D - M
    x = np.zeros(lap2D.shape[0])
    for i in range(100):
        x= np.matmul(inv(M),(b-np.matmul(N,x)))
        residual_vector.append(np.linalg.norm(b - np.dot(lap2D,x)))
        error_vector.append(np.linalg.norm(solution - x))
    
    plt.semilogy([i for i in range(100)], residual_vector)
    plt.xlabel('Iteration number')
    plt.ylabel('Residual vector in l2 norm')
    plt.title("Residual norm in semi-logarithmic scale")
    plt.show()

    
    convergence_factor = [error_vector[i+1]/error_vector[i] for i in range(99)]
    plt.plot([i for i in range(99)], convergence_factor)
    plt.xlabel('Iteration number')
    plt.ylabel('Convergence factor value')
    plt.title("Convergence factor for Gauss-Seidel method")
    plt.show()
    
    
    
    
    residual_drop_factor = [residual_vector[i+1]/residual_vector[i] for i in range(99)]
    plt.plot([i for i in range(99)], residual_drop_factor)
    plt.xlabel('Iteration number')
    plt.ylabel('Residual drop factor value')
    plt.title("Residual drop factor for Gauss-Seidel method")
    plt.show()


Poisson2DGaussSeidel(0.1,0.1)





