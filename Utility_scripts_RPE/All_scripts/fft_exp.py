import numpy as np
def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    print(N)
    n = np.arange(N)
    print("n",n)
    k = n.reshape((N, 1))
    #print("k",k)
    M = np.exp(-2j * np.pi * k * n / N)
    #print(M)
    print(M.shape)
    return np.dot(M, x)

print(DFT_slow(np.sin(np.arange(0, 10, 0.5))))
