import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

class QuantumLinearPDESolver:
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
        self.u = np.zeros((nt + 1, nx + 1))
        self.backend = AerSimulator()
        self.max_amplitude = 0.2
        self.measurement_scale = 0.1

    def initial_conditions(self, u0):
        self.u[0, :] = u0

    def boundary_conditions(self, left_bc, right_bc):
        self.u[:, 0] = left_bc
        self.u[:, -1] = right_bc

    def solve(self):
        for n in range(self.nt):
            for i in range(1, self.nx):
                laplacian = (self.u[n, i + 1] - 2 * self.u[n, i] + self.u[n, i - 1]) / self.dx**2
                gradient = (self.u[n, i + 1] - self.u[n, i - 1]) / (2 * self.dx)
                x = self.x_range[0] + i * self.dx
                t = self.t_range[0] + n * self.dt
                quantum_term = self.quantum_update_step(laplacian, gradient, self.u[n, i], self.f(x, t))
                self.u[n + 1, i] = self.u[n, i] + self.dt * (
                    self.a * laplacian +
                    self.b * gradient +
                    self.c * self.u[n, i] +
                    self.f(x, t)
                ) + quantum_term

    def quantum_update_step(self, laplacian, gradient, u_current, f_val):
        qc = QuantumCircuit(3, 1)
        theta = np.arctan(self.max_amplitude * (
            self.a * laplacian +
            self.b * gradient +
            self.c * u_current +
            f_val
        ))
        qc.ry(theta, 0)
        qc.crx(np.pi / 2, 0, 1)
        qc.crz(np.pi / 2, 0, 2)
        qc.barrier()
        qc.h([1, 2])
        qc.measure([2], [0])
        job = self.backend.run(qc, shots=20)
        result = job.result()
        counts = result.get_counts()
        return self.measurement_scale * (counts.get('0', 0) - counts.get('1', 0)) / 100

    def get_solution(self):
        return self.u
