import numpy as np

def cond_num_matrix(n, cond_num):
    """
    Generate an n x n random matrix with a specified condition number.
    
    Parameters:
        n (int): Size of the matrix.
        cond_num (float): Desired condition number.
    
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

