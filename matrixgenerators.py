import numpy as np

def cond_num_matrix(n, cond_num):
    """
    Generate an n x n random matrix with a specified condition number.
    
    Parameters:
        n (int): Size of the matrix.
        cond_num (float): condition number.
    
    Returns:
        A (ndarray): n x n matrix with the specified condition number.
    """
    
    singular_values = np.linspace(cond_num, 1, n)  # because condition number is sigma_max / sigma_min
    Sigma = np.diag(singular_values)
    
    # we generate random orthogonal matrices U and V so A=USigmaV^T by SVD.
    U, _ = np.linalg.qr(np.random.randn(n, n))  
    V, _ = np.linalg.qr(np.random.randn(n, n))

    A = U @ Sigma @ V.T  
    
    return A

def make_matrix_sparse(A, sparse_percent):
    """
    Generate an n x n random matrix with a specified condition number.
    
    Parameters:
        A: A random matrix 
        sparse_percent (float): percentage of entries that are 0, between 0 and 1
    
    Returns:
        A_sparse (ndarray): n x n matrix with the specified sparsity.
    """
    # Be mindful, this method enforces sparsity on existing matrices.
    mask = np.random.rand(A.shape[0], A.shape[1]) > sparse_percent 
    A_sparse = A * mask
    return A_sparse

def generate_eigenvalue_range(n, eigen_min, eigen_max):
    """
    Generate a random matrix with eigenvalues uniformly distributed between a given range.
    Less clustered eigenvalues will have a high (eigen_max - eigen_min), more clustereed will have lower.
    
    Parameters:
        n (int): Size of the matrix (n x n).
        eigen_min (float): Minimum eigenvalue.
        eigen_max (float): Maximum eigenvalue.
    
    Returns:
        A (ndarray): Generated n x n matrix with uniformly distributed eigenvalues.
        eigenvalues (ndarray): Eigenvalues used in the matrix.
    """
    if n < 2:
        raise ValueError("Matrix size n must be at least 2 to include eigen_min and eigen_max.")
    
    # generate n-2 random evals including the min and max eigenvalue we set, n in total
    rand_eigenvalues = np.random.uniform(low=eigen_min, high=eigen_max, size=n-2)
    eigenvalues = np.concatenate(([eigen_min], rand_eigenvalues, [eigen_max]))
    Sigma = np.diag(eigenvalues)  # Create diagonal matrix of eigenvalues
    
    # generate random orthogonal matrices, calculate eigendecomp
    V, _ = np.linalg.qr(np.random.randn(n, n))
    A = V @ Sigma @ V.T
    return A, eigenvalues