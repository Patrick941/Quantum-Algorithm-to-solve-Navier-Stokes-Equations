import quantum_linear_pde_solver
import linear_pde_solver
import numpy as np

class HybridNavierStokesSolver:
    def __init__(self, nu, force, x_range, t_range, nx, nt, scale_back=1, use_quantum_solver=True):
        self.nu = nu
        self.force = force
        self.x_range = x_range
        self.t_range = t_range
        self.nx = nx
        self.nt = nt // scale_back  # Reduce the number of iterations by the scale_back factor
        self.dx = (x_range[1] - x_range[0]) / nx
        self.dt = (t_range[1] - t_range[0]) / self.nt
        self.u = np.zeros((self.nt + 1, nx + 1))  # Velocity field
        self.p = np.zeros((self.nt + 1, nx + 1))  # Pressure field
        self.use_quantum_solver = use_quantum_solver

    def initial_conditions(self, u0):
        self.u[0, :] = u0

    def boundary_conditions(self, u_left, u_right, p_left, p_right):
        self.u[:, 0] = u_left(self.t_range)
        self.u[:, -1] = u_right(self.t_range)
        self.p[:, 0] = p_left(self.t_range)
        self.p[:, -1] = p_right(self.t_range)

    def get_solver_class(self):
        if self.use_quantum_solver:
            return quantum_linear_pde_solver.QuantumLinearPDESolver
        else:
            return linear_pde_solver.LinearPDESolver

    def solve_diffusion(self):
        # Solve the diffusion equation for velocity
        SolverClass = self.get_solver_class()
        diffusion_solver = SolverClass(
            a=self.nu, b=0, c=0, f=lambda x, t: 0,
            x_range=self.x_range, t_range=self.t_range,
            nx=self.nx, nt=self.nt
        )
        diffusion_solver.initial_conditions(self.u[0, :])
        diffusion_solver.boundary_conditions(self.u[:, 0], self.u[:, -1])
        diffusion_solver.solve()
        self.u = diffusion_solver.get_solution()

    def solve_poisson(self):
        # Solve the Poisson equation for pressure
        source_term = self.compute_source_term()
        SolverClass = self.get_solver_class()
        poisson_solver = SolverClass(
            a=1, b=0, c=0, f=lambda x, t: source_term[int(t / self.dt), int(x / self.dx)],
            x_range=self.x_range, t_range=self.t_range,
            nx=self.nx, nt=self.nt
        )
        poisson_solver.initial_conditions(self.p[0, :])
        poisson_solver.boundary_conditions(self.p[:, 0], self.p[:, -1])
        poisson_solver.solve()
        self.p = poisson_solver.get_solution()

    def compute_source_term(self):
        # Compute the source term for the Poisson equation
        source_term = np.zeros((self.nt + 1, self.nx + 1))
        for n in range(self.nt):
            for i in range(1, self.nx):
                laplacian_u = (self.u[n, i + 1] - 2 * self.u[n, i] + self.u[n, i - 1]) / self.dx**2
                gradient_u = (self.u[n, i + 1] - self.u[n, i - 1]) / (2 * self.dx)
                x = self.x_range[0] + i * self.dx
                t = self.t_range[0] + n * self.dt
                source_term[n, i] = gradient_u - self.nu * laplacian_u + self.force(x, t)
        return source_term

    def solve(self):
        for n in range(self.nt):
            print(f"Solving time step {n + 1}/{self.nt}")
            self.solve_diffusion()
            self.solve_poisson()