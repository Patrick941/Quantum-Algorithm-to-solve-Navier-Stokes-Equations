import numpy as np

class NonLinearPDESolver:
    def __init__(self, nx, ny, nt, dt, rho, nu, x_range=(0, 2), y_range=(0, 2)):
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.dt = dt
        self.rho = rho
        self.nu = nu
        self.x_range = x_range
        self.y_range = y_range

        self.dx = (x_range[1] - x_range[0]) / (nx - 1)
        self.dy = (y_range[1] - y_range[0]) / (ny - 1)

        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        self.b = np.zeros((ny, nx))

        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        self.X, self.Y = np.meshgrid(x, y)

    def apply_boundary_conditions(self):
        self.u[0, :] = 0
        self.u[-1, :] = 0
        self.v[0, :] = 0
        self.v[-1, :] = 0

        self.u[:, 0] = 1
        self.v[:, 0] = 0

        self.u[:, -1] = self.u[:, -2]
        self.v[:, -1] = self.v[:, -2]

        self.p[0, :] = self.p[1, :]
        self.p[-1, :] = self.p[-2, :]
        self.p[:, 0] = self.p[:, 1]
        self.p[:, -1] = 0

        obstacle_mask = (self.X - 1.0)**2 + (self.Y - 1.0)**2 <= 0.2**2
        self.u[obstacle_mask] = 0
        self.v[obstacle_mask] = 0
        
    def add_obstacle(self, center_x, center_y, radius):
        obstacle_mask = (self.X - center_x)**2 + (self.Y - center_y)**2 <= radius**2
        self.u[obstacle_mask] = 0
        self.v[obstacle_mask] = 0

    def pressure_poisson(self, num_iterations=50):
        pn = np.empty_like(self.p)
        for _ in range(num_iterations):
            pn = self.p.copy()
            self.p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * self.dy**2 + (pn[2:, 1:-1] + pn[:-2, 1:-1]) * self.dx**2) / (2 * (self.dx**2 + self.dy**2)) - self.dx**2 * self.dy**2 / (2 * (self.dx**2 + self.dy**2)) * self.b[1:-1, 1:-1])

            obstacle_mask = (self.X - 1.0)**2 + (self.Y - 1.0)**2 <= 0.2**2
            self.p[obstacle_mask] = 0

            self.apply_boundary_conditions()

    def compute_source_term(self):
        gravity = 0.1
        self.b[1:-1, 1:-1] = (self.rho * (1 / self.dt * ((self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2 * self.dx) + (self.v[2:, 1:-1] - self.v[:-2, 1:-1]) / (2 * self.dy)) - ((self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2 * self.dx))**2 - 2 * ((self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2 * self.dy) * (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2 * self.dx)) - ((self.v[2:, 1:-1] - self.v[:-2, 1:-1]) / (2 * self.dy))**2) + gravity)

    def update_velocity(self):
        un = self.u.copy()
        vn = self.v.copy()

        self.u[1:-1, 1:-1] = (un[1:-1, 1:-1] - un[1:-1, 1:-1] * self.dt / self.dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) - vn[1:-1, 1:-1] * self.dt / self.dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) - self.dt / (2 * self.rho * self.dx) * (self.p[1:-1, 2:] - self.p[1:-1, :-2]) + self.nu * (self.dt / self.dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) + self.dt / self.dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))

        self.v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - un[1:-1, 1:-1] * self.dt / self.dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) - vn[1:-1, 1:-1] * self.dt / self.dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) - self.dt / (2 * self.rho * self.dy) * (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) + self.nu * (self.dt / self.dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) + self.dt / self.dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))

        self.apply_boundary_conditions()
        self.add_obstacle(center_x=1.0, center_y=1.0, radius=0.2)

    def solve(self):
        for _ in range(self.nt):
            self.compute_source_term()
            self.pressure_poisson()
            self.update_velocity()

    def get_velocity_field(self):
        return self.u, self.v
