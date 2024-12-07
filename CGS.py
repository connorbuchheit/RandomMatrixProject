import numpy as np

def solve_cgs(A, b, x_0=None, num_iter=int(1e5), tol=1e-9):
    '''
    Ax=b iterative solver CGS (Conjugate Gradient Squared Method)
    Inputs: 
    Matrix A (square, non-symmetric, n x n)
    Vector b (n x 1)
    Optional: 
    x_0 (initial guess)
    max_iter: maximum number of iterations, adjust as necessary
    tolerance tol: terminate if norm of residual is small enough 
    Outputs: 
    Vector x (n x 1)
    Bool converged (info on whether we converged to desired tol)
    Implemented via paper "How Fast Are Nonsymmetric Matrix Iterations" 
    by Nachtigal, Reddy, and Trefethen.
    '''
    n = len(b)
    if x_0 is None:
        x = np.zeros(n) # initial guess all zeroes if no initial guess 
    else:
        if len(x_0) != n: # ensure correct shape
            raise ValueError(f"Initial guess of wrong dimension: {len(x_0)} instead of {n}")
        x = x_0.copy()
    if A.shape[0] != n or A.shape[1] != n: # ensure shapes of matrices are fine
        raise ValueError(f"A is of the wrong dimension: {A.shape[0]} x {A.shape[1]}, not {n} x {n}.")
    
   
    r = b - A @ x 
    r_tilde = r.copy() 
    q = np.zeros(n); p = np.zeros(n); rho = 1
    for _ in range(num_iter): # following steps in paper
        rho_old = rho
        rho = np.vdot(r_tilde, r)
        beta = rho / rho_old 
        u = r + beta * q
        p = u + beta * (q + beta * p) 
        v = A @ p 
        sigma = np.vdot(r_tilde, v)
        if abs(sigma) < 1e-12:
            print("Warning: Sigma is too small. Restarting iteration.")
            q = np.zeros_like(q)
            p = np.zeros_like(p)
            continue
        alpha = rho / sigma
        q = u - alpha * v
        r = r - alpha * A @ (u + q) 
        x = x + alpha * (u + q)
        # check convergence
        res_norm = np.linalg.norm(r)
        if res_norm < tol:
            print("Converged early.")
            return x
    return x 


n = 100
A = np.random.rand(n,n)
A[0,1] += 1
A *= 100
print(np.linalg.cond(A))
b = np.random.rand(n)

x_cgs = solve_cgs(A, b)

x_bicg, info = bicg(A, b)

# Solve using NumPy's direct solver
x_direct = np.linalg.solve(A, b)

print("Solution using NumPy direct solver:\n", x_direct)