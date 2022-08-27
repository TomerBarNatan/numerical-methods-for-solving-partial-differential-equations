def damped_jacobi(A, b, x, iteration_number, omega):
    D = A[0, 0]
    for i in range(iteration_number):
        x = x +  (omega / D) * (b - A @ x)
    return x