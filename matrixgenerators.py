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
    mask = np.random.rand(A.shape) > sparse_percent 
    A_sparse = A * mask
    return A_sparse
