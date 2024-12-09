import numpy as np
from IPython.display import clear_output

def solve_cgn(A, b, max_iter = int(1e5), x0 = None, tol = 1e-12):
    '''
    Iterative Conjugate Gradient on Normal equations solver of Ax = b

    Parameters:
        INPUTS: 
        A : (N x N) nonsymmetric matrix
        b : (N x 1) vector
        num_iter (optional) : Number of iterations before forced termination
        x0 : (N x 1) initial guess vector 
        tol (optional) : residual tolerance for approximating convergence 

        OUTPUT: 
        x : (N x 1) vector solution to Ax = b and A* Ax = A* b (multiplication by conjugate to achieve PSD symmetric matrix A* A)
        res : vector of residual norms 
        path : trajectory of steps from `x_0` to x_final
        conv : boolean indicating convergence within `num_iter` iterations

    Implemented via paper "How Fast Are Nonsymmetric Matrix Iterations" 
    by Nachtigal, Reddy, and Trefethen.
    '''

    # Ensure A and b are NumPy arrays
    A = np.array(A)
    b = np.array(b)
    m, n = A.shape

    # Initialize variables
    if x0 is None:
        x = np.zeros(n)  # Initial guess
    else:
        x = np.array(x0)

    r = A.T @ (b - A @ x)  # Initial residual, this seems to work better than the original paper (?)
    p = r.copy()  # Initial search direction (just initialize to r rather than 0)
    residuals = [np.linalg.norm(r)]
    converged = False # this indicates that the iteration converged in less than max_iter steps

    for k in range(max_iter):
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(Ap, Ap)  # Step size
        x = x + alpha * p  # Update solution
        r_new = r - alpha * (A.T @ Ap)  # Update residual

        # Check for convergence
        residual_norm = np.linalg.norm(r_new)
        residuals.append(residual_norm)
        if residual_norm < tol:
            converged = True
            break

        beta = np.dot(r_new, r_new) / np.dot(r, r)  # Update factor
        p = r_new + beta * p  # Update search direction
        r = r_new  # Update residual

    return x, residuals, converged