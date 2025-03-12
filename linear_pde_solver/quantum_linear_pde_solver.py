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
        self.u = np.zeros((nt+1, nx+1))
        self.backend = AerSimulator(method="statevector")

    def initial_conditions(self, u0):
        self.u[0, :] = u0

    def boundary_conditions(self, left_bc, right_bc):
        self.u[:, 0] = left_bc
        self.u[:, -1] = right_bc

    def solve(self):
        for n in range(self.nt):
            for i in range(1, self.nx):
                self.u[n+1, i] = self.run_quantum_circuit(self.u[n, i-1], self.u[n, i], self.u[n, i+1])

    def run_quantum_circuit(self, u_left, u_center, u_right):
        qc = QuantumCircuit(3)
        values = np.array([u_left, u_center, u_right], dtype=complex)
        norm = np.linalg.norm(values)
        if norm < 1e-12:
            values = np.zeros_like(values)
            values[0] = 1.0
        else:
            values /= norm
        init_state = np.zeros(8, dtype=complex)
        init_state[0] = values[0]
        init_state[1] = values[1]
        init_state[2] = values[2]
        qc.initialize(init_state, [0, 1, 2])
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.rx(self.a * self.dt / self.dx**2, 0)
        qc.ry(self.b * self.dt / (2*self.dx), 1)
        qc.rz(self.c * self.dt, 2)
        qc.save_statevector()
        result = self.backend.run(qc).result()
        final_state = result.get_statevector(qc)
        return (final_state[1] * norm).real


    def get_solution(self):
        return self.u
