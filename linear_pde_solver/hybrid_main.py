from hybrid_non_linear_pde_solver import HybridNavierStokesSolver
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Define parameters
nu = 0.1  # Kinematic viscosity
force = lambda x, t: np.sin(np.pi * x)  # External force
x_range = (0, 1)
t_range = (0, 1)
nx = 10
nt = 10
scale_back = 1  # Adjust this as needed

# Initial and boundary conditions
amplitude = 20  # Increase the amplitude to increase the velocity
u0 = amplitude * np.sin(np.pi * np.linspace(x_range[0], x_range[1], nx + 1))
u_left = lambda t: 0
u_right = lambda t: 0
p_left = lambda t: 1 + t[0]  # Introduce a time-dependent pressure gradient
p_right = lambda t: t[0]

# Create solvers for quantum and classical methods
quantum_solver = HybridNavierStokesSolver(nu, force, x_range, t_range, nx, nt, scale_back, use_quantum_solver=True)
classical_solver = HybridNavierStokesSolver(nu, force, x_range, t_range, nx, nt, scale_back, use_quantum_solver=False)

# Set initial and boundary conditions for both solvers
for solver in [quantum_solver, classical_solver]:
    solver.initial_conditions(u0)
    solver.boundary_conditions(u_left, u_right, p_left, p_right)

# Solve using quantum method
quantum_solver.solve()
quantum_velocity_field = quantum_solver.u
quantum_pressure_field = quantum_solver.p

# Solve using classical method
classical_solver.solve()
classical_velocity_field = classical_solver.u
classical_pressure_field = classical_solver.p

# Create a meshgrid for plotting
x = np.linspace(x_range[0], x_range[1], nx + 1)
t = np.linspace(t_range[0], t_range[1], nt + 1)
X, T = np.meshgrid(x, t)

# Determine the limits for the plots
vmin_velocity = min(quantum_velocity_field.min(), classical_velocity_field.min())
vmax_velocity = max(quantum_velocity_field.max(), classical_velocity_field.max())
vmin_pressure = min(quantum_pressure_field.min(), classical_pressure_field.min())
vmax_pressure = max(quantum_pressure_field.max(), classical_pressure_field.max())

# Plot the velocity and pressure fields side by side in a 2x2 grid
fig = plt.figure(figsize=(14, 12))

# Plot the quantum velocity field
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot_surface(X, T, quantum_velocity_field.T, cmap='viridis', vmin=vmin_velocity, vmax=vmax_velocity)
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_zlabel('Velocity')
ax1.set_title('Quantum Velocity Field')
ax1.set_zlim(vmin_velocity, vmax_velocity)

# Plot the quantum pressure field
ax2 = fig.add_subplot(222, projection='3d')
ax2.plot_surface(X, T, quantum_pressure_field.T, cmap='viridis', vmin=vmin_pressure, vmax=vmax_pressure)
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('Pressure')
ax2.set_title('Quantum Pressure Field')
ax2.set_zlim(vmin_pressure, vmax_pressure)

# Plot the classical velocity field
ax3 = fig.add_subplot(223, projection='3d')
ax3.plot_surface(X, T, classical_velocity_field.T, cmap='viridis', vmin=vmin_velocity, vmax=vmax_velocity)
ax3.set_xlabel('x')
ax3.set_ylabel('t')
ax3.set_zlabel('Velocity')
ax3.set_title('Classical Velocity Field')
ax3.set_zlim(vmin_velocity, vmax_velocity)

# Plot the classical pressure field
ax4 = fig.add_subplot(224, projection='3d')
ax4.plot_surface(X, T, classical_pressure_field.T, cmap='viridis', vmin=vmin_pressure, vmax=vmax_pressure)
ax4.set_xlabel('x')
ax4.set_ylabel('t')
ax4.set_zlabel('Pressure')
ax4.set_title('Classical Pressure Field')
ax4.set_zlim(vmin_pressure, vmax_pressure)

plt.tight_layout()
plt.savefig('hybrid_navier_stokes_solution_comparison.png')
