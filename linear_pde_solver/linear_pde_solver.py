import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class LinearPDESolver:
    def __init__(self, N):
        self.N = N
        self.A, self.b = self.discretize()

    def discretize(self):
        h = 1.0 / (self.N + 1)
        diag_main = 2.0 * np.ones(self.N) / (h * h)
        diag_off = -1.0 * np.ones(self.N - 1) / (h * h)
        A = (np.diag(diag_main, 0)
             + np.diag(diag_off, 1)
             + np.diag(diag_off, -1))
        b = np.ones(self.N)
        return A, b

    def solve(self):
        A_sparse = sp.csr_matrix(self.A)
        x = spla.spsolve(A_sparse, self.b)
        return x