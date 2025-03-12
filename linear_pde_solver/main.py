import numpy as np
from scipy.integrate import solve_ivp
import linear_pde_solver
import quantum_linear_pde_solver
import matplotlib.pyplot as plt

alpha = 0.01
x_range = (0, 1)
t_range = (0, 1)
nx = 10
nt = 100

x = np.linspace(x_range[0], x_range[1], nx + 1)
dx = x[1] - x[0]
dt = (t_range[1] - t_range[0]) / nt

u0 = np.sin(np.pi * x)

left_bc = 0
right_bc = 0

def heat_equation(t, u):
    dudx = np.zeros_like(u)
    dudx[1:-1] = alpha * (u[:-2] - 2 * u[1:-1] + u[2:]) / dx**2
    return dudx

sol = solve_ivp(heat_equation, t_range, u0, t_eval=np.linspace(t_range[0], t_range[1], nt + 1))
solution = sol.y

solver = linear_pde_solver.LinearPDESolver(alpha, 0, 0, lambda x, t: 0, x_range, t_range, nx, nt)
solver.initial_conditions(u0)
solver.boundary_conditions(left_bc, right_bc)
solver.solve()
your_solution = solver.get_solution()

quantum_solver = quantum_linear_pde_solver.QuantumLinearPDESolver(alpha, 0, 0, lambda x, t: 0, x_range, t_range, nx, nt)
quantum_solver.initial_conditions(u0)
quantum_solver.boundary_conditions(left_bc, right_bc)
quantum_solver.solve()
quantum_solution = quantum_solver.get_solution()

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

for i in range(0, nt + 1, nt // 10):
    axs[0].plot(x, solution[:, i], label=f'SciPy t={sol.t[i]:.2f}')

for i in range(0, nt + 1, nt // 10):
    axs[1].plot(x, your_solution[i, :], '--', label=f'Your Solver t={sol.t[i]:.2f}')

for i in range(0, nt + 1, nt // 10):
    axs[2].plot(x, quantum_solution[i, :], '--', label=f'Quantum Solver t={sol.t[i]:.2f}')

axs[0].set_title('SciPy Solution')
axs[1].set_title('Your Solver Solution')
axs[2].set_title('Quantum Solver Solution')

for ax in axs:
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.legend()

plt.tight_layout()
plt.savefig('Images/comparison_solution.png')
