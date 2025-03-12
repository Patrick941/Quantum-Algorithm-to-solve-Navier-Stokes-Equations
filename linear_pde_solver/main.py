import numpy as np
from scipy.integrate import solve_ivp
import linear_pde_solver
import quantum_linear_pde_solver
import matplotlib.pyplot as plt
import non_linear_pde_solver

# Parameters for the heat equation
alpha = 0.01
x_range = (0, 1)
t_range = (0, 0.1)
nx = 10
nt = 100

x = np.linspace(x_range[0], x_range[1], nx + 1)
dx = x[1] - x[0]
dt = (t_range[1] - t_range[0]) / nt

u0 = np.sin(np.pi * x)

left_bc = 0
right_bc = 0

# Heat equation function for SciPy's solve_ivp
def heat_equation(t, u):
    dudx = np.zeros_like(u)
    dudx[1:-1] = alpha * (u[:-2] - 2 * u[1:-1] + u[2:]) / dx**2
    return dudx

# Solve using SciPy's solve_ivp
sol = solve_ivp(heat_equation, t_range, u0, t_eval=np.linspace(t_range[0], t_range[1], nt + 1))
solution = sol.y

# Solve using the LinearPDESolver
solver = linear_pde_solver.LinearPDESolver(alpha, 0, 0, lambda x, t: 0, x_range, t_range, nx, nt)
solver.initial_conditions(u0)
solver.boundary_conditions(left_bc, right_bc)
solver.solve()
your_solution = solver.get_solution()

# Solve using the QuantumLinearPDESolver
quantum_solver = quantum_linear_pde_solver.QuantumLinearPDESolver(alpha, 0, 0, lambda x, t: 0, x_range, t_range, nx, nt)
quantum_solver.initial_conditions(u0)
quantum_solver.boundary_conditions(left_bc, right_bc)
quantum_solver.solve()
quantum_solution = quantum_solver.get_solution()

# Initialize the solver
navier_stokes_solver = non_linear_pde_solver.NonLinearPDESolver(nx=41, ny=41, nt=500, dt=0.001, rho=1, nu=0.1)

# Add a circular obstacle at the center
center_x = 1.0  # X-coordinate of the center
center_y = 1.0  # Y-coordinate of the center
radius = 0.2    # Radius of the obstacle
navier_stokes_solver.add_obstacle(center_x, center_y, radius)

# Solve the Navier-Stokes equations
navier_stokes_solver.solve()
navier_stokes_solution = navier_stokes_solver.get_velocity_field()

# Plotting
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# SciPy solution
for i in range(0, nt + 1, nt // 10):
    axs[0, 0].plot(x, solution[:, i], label=f'SciPy t={sol.t[i]:.2f}')

# Your solver solution
for i in range(0, nt + 1, nt // 10):
    axs[1, 0].plot(x, your_solution[i, :], '--', label=f'Your Solver t={sol.t[i]:.2f}')

# Quantum solver solution
for i in range(0, nt + 1, nt // 10):
    axs[2, 0].plot(x, quantum_solution[i, :], '--', label=f'Quantum Solver t={sol.t[i]:.2f}')

# Navier-Stokes solver solution (plotting velocity magnitude)
x_ns = np.linspace(0, 2, 41)
y_ns = np.linspace(0, 2, 41)
u, v = navier_stokes_solution
speed = np.sqrt(u**2 + v**2)
contour = axs[1, 1].contourf(x_ns, y_ns, speed, levels=50, cmap='viridis')
fig.colorbar(contour, ax=axs[1, 1])
axs[1, 1].set_title('Navier-Stokes Solver Velocity Magnitude')

# Set titles and labels
axs[0, 0].set_title('SciPy Solution')
axs[1, 0].set_title('Your Solver Solution')
axs[2, 0].set_title('Quantum Solver Solution')

for ax in axs[:, 0]:
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('y')

plt.tight_layout()
plt.savefig('Images/comparison_solution.png')