import numpy as np

# Does not work entirely — most finicky of the bunch.

def solve_qmr(A, b, x_0=None, num_iter = int(1e5), tol=1e-6):
    '''
    Solve Ax=b using Quasi-Minimal Residual (QMR) method.

    Parameters:
        INPUTS: 
        A : (N x N) nonsymmetric matrix
        b : (N x 1) vector
        num_iter (optional) : Number of iterations before forced termination
        x_0 : (N x 1) initial guess vector 
        tol (optional) : residual tolerance for approximating convergence 

        OUTPUT: 
        x : (N x 1) vector solution to Ax = b
        Implemented from QMR: a quasi-minimal residual method for non-Hermitian linear systems
        by Freund and Nachtigal
    '''
    n = len(b)
    if x_0 is None:
        x = np.zeros(n)
    else:
        x = x_0.copy()
    if A.shape[0] != n or A.shape[1] != n: # ensure shapes of matrices are fine
        raise ValueError(f"A is of the wrong dimension: {A.shape[0]} x {A.shape[1]}, not {n} x {n}.")

    r = b - A @ x # initial residual
    r_tilde = r.copy() # first residual
    rho_prev = np.vdot(r_tilde, r)
    v = r.copy()
    p = np.zeros_like(r); w = np.zeros_like(r); q = np.zeros_like(r)

    for _ in range(num_iter):
        v = r / rho_prev
        w = r_tilde / rho_prev # Lanczos biorthogonalization step
        alpha = np.dot(w, A @ v)
        u = r - alpha * v
        beta = np.dot(w, A @ u)
        if abs(beta) < tol or abs(alpha) < tol:
            print("Warning: Too small values")
        p = v + beta * p
        q = w + beta * q 
        x = x + alpha * p
        r = r - alpha * (A @ p)
        r_tilde = r_tilde - alpha * A.T @ q
        rho = np.dot(r_tilde, r)

        # check convergence 
        if np.linalg.norm(r) < tol:
            print("Converged early")
            return x 
        rho_prev = rho
    print("Didn't converge :(")
    return x

n = 100
A = np.random.rand(n,n)
A[0,1] += 1
A *= 100
print(np.linalg.cond(A))
b = np.random.rand(n)

x_cgs = solve_qmr(A, b)

# Solve using NumPy's direct solver
x_direct = np.linalg.solve(A, b)

print("Solution using NumPy direct solver:\n", x_direct)