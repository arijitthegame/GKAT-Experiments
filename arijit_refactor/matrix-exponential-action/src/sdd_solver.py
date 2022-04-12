import numpy as np

'''
Adding the Jacobi and the Gauss-Siedel method. They only work for (strict) diagonally dominant matrices
'''

#TODO: Add more sophisicated solvers. And torchify these functions. 

def jacobi(A, b, tolerance=1e-10, max_iterations=10000):

    x = np.zeros_like(b, dtype=np.double)

    T = A - np.diag(np.diagonal(A))

    for k in range(max_iterations):

        x_old  = x.copy()

        x[:] = (b - np.dot(T, x)) / np.diagonal(A)

        if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tolerance:
            break

    return x


def gauss_seidel(A, b, tolerance=1e-10, max_iterations=10000):

    x = np.zeros_like(b, dtype=np.double)

    #Iterate
    for k in range(max_iterations):

        x_old  = x.copy()

        #Loop over rows
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_old[(i+1):])) / A[i ,i]

        #Stop condition
        if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tolerance:
            break

    return x

# test
if name == '__main__':
    from scipy.linalg import solve

    A = A = np.random.rand(200,200)
    A1 = A + A.transpose() + 200*np.identity(200)
    b = np.random.rand(200)
    x = np.random.rand(200)
    y = jacobi(A1,b,x, 2000)
    y1 = solve(A1,b)
    print("Jacobi approx solution:", y)
    print("Actual Solution :", y1) 
