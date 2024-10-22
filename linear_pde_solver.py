import numpy as np
from scipy.linalg import solve_banded

class LinearPDESolver:
    def __init__(self, a, b, c, f, x_range, t_range, nx, nt):
        self.a = a
        self.b = b
        self.c = c
        self.f = f
        self.x_range = x_range
        self.t_range = t_range
        self.nx = nx
        self.nt = nt
        self.dx = (x_range[1] - x_range[0]) / nx
        self.dt = (t_range[1] - t_range[0]) / nt
        self.u = np.zeros((nt+1, nx+1))

    def initial_conditions(self, u0):
        self.u[0, :] = u0

    def boundary_conditions(self, left_bc, right_bc):
        self.u[:, 0] = left_bc
        self.u[:, -1] = right_bc

    def solve(self):
        for n in range(0, self.nt):
            for i in range(1, self.nx):
                self.u[n+1, i] = (self.u[n, i] + self.dt * (
                    self.a * (self.u[n, i+1] - 2*self.u[n, i] + self.u[n, i-1]) / self.dx**2 +
                    self.b * (self.u[n, i+1] - self.u[n, i-1]) / (2*self.dx) +
                    self.c * self.u[n, i] +
                    self.f(self.x_range[0] + i*self.dx, self.t_range[0] + n*self.dt)
                ))

    def get_solution(self):
        return self.u