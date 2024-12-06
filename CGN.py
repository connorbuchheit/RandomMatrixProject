import numpy as np
from IPython.display import clear_output

def solve_cgn(A, b, num_iter = int(1e5), x_0 = None, tol = 1e-12):
    '''
    Iterative Conjugate Gradient on Normal equations solver of Ax = b

    INPUTS: 
    A : (N x N) nonsymmetric matrix
    b : (N x 1) vector
    num_iter (optional) : Number of iterations before forced termination
    x_0 : (N x 1) initial guess vector 
    tol (optional) : residual tolerance for approximating convergence 

    OUTPUT: 
    x : (N x 1) vector solution to Ax = b
    res : vector of residual norms 
    path : trajectory of steps from `x_0` to x_final
    conv : boolean indicating convergence within `num_iter` iterations

    Implemented via paper "How Fast Are Nonsymmetric Matrix Iterations" 
    by Nachtigal, Reddy, and Trefethen.
    '''

    A = np.matrix(A)
    x_0 = np.zeros(shape = (np.shape(A)[1] , ))

    beta = [0]
    p = [np.zeros_like(b)]
    r = [1e5 * np.ones(shape = (np.shape(A)[0], ))]
    alpha = []
    x = [x_0]
    t = 0

    while np.linalg.norm(np.array(r[-1]).reshape(len(r[-1]), )) >= tol:   
        if t >= num_iter:
            break
        p.append(np.array(A.H @ (np.array(r[-1]).reshape(len(r[-1]), )) + beta[-1] * (np.array(p[-1]).reshape(len(p[-1]), ))).reshape(p[0].shape[0], ))
        alpha.append(np.linalg.norm(A.H @ np.array(r[-1]).reshape(len(r[-1]), ))**2 / np.linalg.norm(A @ np.array(p[-1]).reshape(len(p[-1]), ))**2)
        x.append(np.array(x[-1]).reshape(len(x[-1]), ) + alpha[-1] * np.array(p[-1]).reshape(len(p[-1]), ))
        r.append(np.array(np.array(r[-1]).reshape(len(r[-1]), ) - alpha[-1] * (A @ np.array(p[-1]).reshape(len(p[-1]), ))).reshape(r[0].shape[0], ))
        beta.append(np.linalg.norm(A.H @ np.array(r[-1]).reshape(len(r[-1]), ))**2 / np.linalg.norm(A @ np.array(r[-2]).reshape(len(r[-2]), ))**2)

        if t % 100 == 0:
            print(f'Iteration: {t} \t Error: {np.linalg.norm(np.array(r[-1]).reshape(len(r[-1]), ))}')
        if t % 500 == 0:
            clear_output(wait = True)
        t += 1
    
    if t < num_iter:
        print(f"CGN converged within {num_iter} iterations  :)")
    else:
        print(f"CGN did not converge within {num_iter} iterations  :(")
    
    return x[-1], np.linalg.norm(np.array(r), axis = 1), x, t < num_iter