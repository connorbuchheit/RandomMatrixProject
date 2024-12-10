import numpy as np
import time
from scipy.sparse.linalg import spilu
from scipy.sparse import csr_matrix, csc_matrix

def solve_cgs_ilu(A, b, x_0=None, num_iter=int(1e5), tol=1e-9):
    '''
    Ax=b iterative solver CGS (Conjugate Gradient Squared Method)
    Parameters:
        Inputs: 
        Matrix A (square, non-symmetric, n x n)
        Vector b (n x 1)
        Optional: 
        x_0 (initial guess)
        max_iter: maximum number of iterations, adjust as necessary
        tolerance tol: terminate if norm of residual is small enough 

        Outputs: 
        Vector x (n x 1)
        Vector of residuals
        Converged (Boolean) 
    Implemented via paper "How Fast Are Nonsymmetric Matrix Iterations" 
    by Nachtigal, Reddy, and Trefethen.
    '''
    # Convert A to a sparse matrix if it's not already
    A = csr_matrix(A)  # ADDED: Ensure A is in sparse format for ILU

    # Compute ILU preconditioner
    ilu = spilu(A)  # ADDED: ILU decomposition
    M_inv = lambda v: ilu.solve(v)  # ADDED: Define preconditioner as a function

    n = len(b)
    converged = False
    if x_0 is None:
        x = np.zeros(n)  # Initial guess all zeroes if no initial guess
    else:
        if len(x_0) != n:  # Ensure correct shape
            raise ValueError(f"Initial guess of wrong dimension: {len(x_0)} instead of {n}")
        x = x_0.copy()
    if A.shape[0] != n or A.shape[1] != n:  # Ensure shapes of matrices are fine
        raise ValueError(f"A is of the wrong dimension: {A.shape[0]} x {A.shape[1]}, not {n} x {n}.")

    # Apply preconditioning to the initial residual
    r = M_inv(b - A @ x)  # MODIFIED: Preconditioned residual
    r_tilde = r.copy()  # Initial shadow residual
    q = np.zeros(n)
    p = np.zeros(n)
    rho = 1
    residuals = [np.linalg.norm(r)]

    for _ in range(num_iter):  # Following steps in paper
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
        r = r - M_inv(alpha * A @ (u + q))  # MODIFIED: Preconditioned residual update
        x = x + alpha * (u + q)

        # Check convergence
        res_norm = np.linalg.norm(r)
        residuals.append(res_norm)
        if res_norm < tol:
            converged = True
            return [x, residuals, converged]
    return [x, residuals, converged]



def solve_cgs_jacobi(A, b, x_0=None, num_iter=int(1e5), tol=1e-9):
    '''
    Ax=b iterative solver CGS (Conjugate Gradient Squared Method) with Jacobi Preconditioning.
    Parameters:
        Inputs: 
        Matrix A (square, non-symmetric, n x n)
        Vector b (n x 1)
        Optional: 
        x_0 (initial guess)
        max_iter: maximum number of iterations, adjust as necessary
        tolerance tol: terminate if norm of residual is small enough 

        Outputs: 
        Vector x (n x 1)
        Vector of residuals
        Converged (Boolean) 
    Implemented via paper "How Fast Are Nonsymmetric Matrix Iterations" 
    by Nachtigal, Reddy, and Trefethen.
    '''
    n = len(b)
    converged = False
    if x_0 is None:
        x = np.zeros(n)  # initial guess all zeroes if no initial guess 
    else:
        if len(x_0) != n:  # ensure correct shape
            raise ValueError(f"Initial guess of wrong dimension: {len(x_0)} instead of {n}")
        x = x_0.copy()
    if A.shape[0] != n or A.shape[1] != n:  # ensure shapes of matrices are fine
        raise ValueError(f"A is of the wrong dimension: {A.shape[0]} x {A.shape[1]}, not {n} x {n}.")
    
    # === New: Compute Jacobi preconditioner ===
    M = np.diag(A.T @ A)  # Diagonal of A^T A
    if np.any(M == 0):
        raise ValueError("Jacobi preconditioner contains zero values.")
    M_inv = 1.0 / M       # Inverse of diagonal elements

    # === Initialize residuals with preconditioning ===
    r = b - A @ x 
    r_tilde = r.copy() 
    z = M_inv * r         # Preconditioned residual
    z_tilde = M_inv * r_tilde  # Preconditioned shadow residual
    q = np.zeros(n); p = np.zeros(n); rho = 1
    residuals = [np.linalg.norm(r)]
    
    for _ in range(num_iter):  # following steps in paper
        rho_old = rho
        rho = np.vdot(z_tilde, z)  # Use preconditioned residual here
        beta = rho / rho_old 
        u = z + beta * q
        p = u + beta * (q + beta * p) 
        v = A @ p 
        sigma = np.vdot(z_tilde, v)  # Use preconditioned shadow residual here
        if abs(sigma) < 1e-12:
            print("Warning: Sigma is too small. Restarting iteration.")
            q = np.zeros_like(q)
            p = np.zeros_like(p)
            continue
        alpha = rho / sigma
        q = u - alpha * v
        r = r - alpha * A @ (u + q) 
        z = M_inv * r  # Update preconditioned residual
        x = x + alpha * (u + q)
        
        # Check convergence
        res_norm = np.linalg.norm(r)
        residuals.append(res_norm)
        if res_norm < tol:
            converged = True
            return [x, residuals, converged]
    return [x, residuals, converged]

def solve_cgn_ilu(A, b, max_iter=int(1e5), x0=None, tol=1e-12):
    '''
    Iterative Conjugate Gradient on Normal equations solver of Ax = b with ILU preconditioning.
    '''

    A = csr_matrix(A)
    b = np.array(b)
    m, n = A.shape

    start = time.time() 
    # Compute ILU preconditioner for A^T A
    ATA = A.T @ A  # Normal equations matrix
    ilu = spilu(ATA)  # ILU decomposition
    M_inv = lambda v: ilu.solve(v)  # Preconditioner function

    if x0 is None:
        x = np.zeros(n)  # Initial guess
    else:
        x = np.array(x0)

    r = A.T @ (b - A @ x)  # Initial residual
    z = M_inv(r)  # Preconditioned residual
    p = z.copy()  # Initial search direction
    residuals = [np.linalg.norm(r)]
    converged = False

    for k in range(max_iter):
        Ap = A @ p
        alpha = np.dot(r, z) / np.dot(Ap, Ap)  # Step size
        x = x + alpha * p  # Update solution
        r_new = r - alpha * (A.T @ Ap)  # Update residual
        z_new = M_inv(r_new)  # Preconditioned residual

        # Check for convergence
        residual_norm = np.linalg.norm(r_new)
        residuals.append(residual_norm)
        if residual_norm < tol:
            converged = True
            break

        beta = np.dot(r_new, z_new) / np.dot(r, z)  # Update factor
        p = z_new + beta * p  # Update search direction
        r = r_new
        z = z_new

    end = time.time()
    duration = end - start

    return x, residuals, converged, duration


def solve_cgn_jacobi(A, b, max_iter=int(1e5), x0=None, tol=1e-12):
    '''
    Iterative Conjugate Gradient on Normal equations solver of Ax = b
    with Jacobi preconditioning.

    Parameters:
        INPUTS:
        A : (m x n) nonsymmetric matrix
        b : (m x 1) vector
        max_iter (optional) : Number of iterations before forced termination
        x0 : (n x 1) initial guess vector 
        tol (optional) : residual tolerance for approximating convergence 

        OUTPUT:
        x : (n x 1) vector solution to Ax = b (using preconditioning on A^T A x = A^T b)
        res : vector of residual norms
        conv : boolean indicating convergence within `max_iter` iterations
        duration : Time taken to solve
    '''

    # Ensure A and b are NumPy arrays
    A = np.array(A)
    b = np.array(b)
    m, n = A.shape

    # Start timing
    start = time.time()

    if x0 is None:
        x = np.zeros(n)  # Initial guess
    else:
        x = np.array(x0)

    # Compute the diagonal preconditioner (Jacobi)
    M = np.diag(A.T @ A)  # Diagonal of A^T A
    M_inv = 1.0 / M       # Inverse of the diagonal elements (Jacobi preconditioner)

    r = A.T @ (b - A @ x)  # Initial residual
    z = M_inv * r          # Preconditioned residual
    p = z.copy()           
    residuals = [np.linalg.norm(r)]
    converged = False

    for k in range(max_iter):
        Ap = A @ p
        alpha = np.dot(r, z) / np.dot(Ap, Ap)  # Step size
        x = x + alpha * p                     
        r_new = r - alpha * (A.T @ Ap)         
        z_new = M_inv * r_new                # Preconditioned residual

        # Check for convergence
        residual_norm = np.linalg.norm(r_new)
        residuals.append(residual_norm)
        if residual_norm < tol:
            converged = True
            break

        beta = np.dot(r_new, z_new) / np.dot(r, z)  # Update factor
        p = z_new + beta * p                        # Update search direction
        r = r_new
        z = z_new

    end = time.time()
    duration = end - start

    return x, residuals, converged, duration


def solve_qmr_ilu(A, b, x_0=None, num_iter=int(1e5), tol=1e-6):
    '''
    Solve Ax=b using Quasi-Minimal Residual (QMR) method with ILU preconditioning.

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
    
    if A.shape[0] != n or A.shape[1] != n:  # Ensure shapes of matrices are fine
        raise ValueError(f"A is of the wrong dimension: {A.shape[0]} x {A.shape[1]}, not {n} x {n}.")
    
    # Convert A to a sparse matrix for ILU preconditioning
    A_sparse = csc_matrix(A)

    # Get the ILU preconditioner (use spilu to compute it)
    ilu = spilu(A_sparse)
    
    # Define a function to apply the preconditioner
    def M_inv(v):
        return ilu.solve(v)
    
    # Initialize residuals and variables
    r = b - A @ x  # Initial residual
    r_tilde = r.copy()  # First residual
    rho_prev = np.vdot(r_tilde, r)
    v = r.copy()
    p = np.zeros_like(r)
    w = np.zeros_like(r)
    q = np.zeros_like(r)

    for _ in range(num_iter):
        # Apply preconditioning to r and r_tilde
        v = M_inv(r)  # Preconditioned v
        w = M_inv(r_tilde)  # Preconditioned w
        
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

        # Check convergence
        if np.linalg.norm(r) < tol:
            print("Converged early")
            return x
    
        rho_prev = rho
    
    print("Didn't converge :(")
    return x

def solve_qmr_jacobi(A, b, x_0=None, num_iter=int(1e5), tol=1e-6):
    '''
    Solve Ax=b using Quasi-Minimal Residual (QMR) method with Jacobi Preconditioning.

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
    
    if A.shape[0] != n or A.shape[1] != n:  # ensure shapes of matrices are fine
        raise ValueError(f"A is of the wrong dimension: {A.shape[0]} x {A.shape[1]}, not {n} x {n}.")

    # Jacobi Preconditioning: D is the diagonal of A
    D_inv = np.diag(1.0 / np.diag(A))  # Inverse of diagonal of A (Jacobi preconditioner)
    
    # Initialize residuals and vectors
    r = b - A @ x  # Initial residual
    r_tilde = r.copy()  # First residual
    rho_prev = np.vdot(r_tilde, r)
    v = r.copy()  # Direction vectors
    p = np.zeros_like(r)
    w = np.zeros_like(r)
    q = np.zeros_like(r)

    for _ in range(num_iter):
        # Apply Jacobi preconditioning: z = D_inv * r
        z = D_inv @ r
        
        v = z / rho_prev
        w = r_tilde / rho_prev  # Biorthogonalization step
        alpha = np.dot(w, A @ v)
        
        u = r - alpha * v
        beta = np.dot(w, A @ u)
        
        if abs(beta) < tol or abs(alpha) < tol:
            print("Warning: Too small values for alpha or beta, iteration might be unstable.")
        
        p = v + beta * p
        q = w + beta * q
        x = x + alpha * p
        r = r - alpha * (A @ p)
        r_tilde = r_tilde - alpha * A.T @ q
        rho = np.dot(r_tilde, r)

        # Check convergence
        if np.linalg.norm(r) < tol:
            print("Converged early")
            return x
    
        rho_prev = rho  # Update rho for the next iteration

    print("Didn't converge :(")
    return x
