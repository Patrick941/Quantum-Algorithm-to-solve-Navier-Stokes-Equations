import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def simulate_fluid_flow(grid_size, time_steps, viscosity, density):
    u = np.ones((grid_size, grid_size)) / np.sqrt(2)  # Initial condition: uniform flow from bottom left to top right
    v = np.ones((grid_size, grid_size)) / np.sqrt(2)
    
    u_list = [u.copy()]
    v_list = [v.copy()]
    
    for _ in range(time_steps):
        laplacian_u = np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) + \
                      np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
        laplacian_v = np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) + \
                      np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4 * v
        
        u += viscosity * laplacian_u / density
        v += viscosity * laplacian_v / density
        
        u_list.append(u.copy())
        v_list.append(v.copy())
    
    return u_list, v_list

def update_quiver(num, Q, u_list, v_list):
    Q.set_UVC(u_list[num], v_list[num])
    return Q,

grid_size = 50
time_steps = 100
viscosity = 0.1
density = 1.0

# Adding a simple object as an obstacle
obstacle = np.zeros((grid_size, grid_size))
obstacle[20:30, 20:30] = 1  # Creating a square obstacle in the middle of the grid

def apply_obstacle(u, v, obstacle):
    u[obstacle == 1] = 0
    v[obstacle == 1] = 0
    return u, v

# Modify the simulation to include the obstacle
def simulate_fluid_flow_with_obstacle(grid_size, time_steps, viscosity, density, obstacle):
    u = np.ones((grid_size, grid_size)) / np.sqrt(2)  # Initial condition: uniform flow from bottom left to top right
    v = np.ones((grid_size, grid_size)) / np.sqrt(2)
    
    u_list = [u.copy()]
    v_list = [v.copy()]
    
    for _ in range(time_steps):
        laplacian_u = np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) + \
                      np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
        laplacian_v = np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) + \
                      np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4 * v
        
        u += viscosity * laplacian_u / density
        v += viscosity * laplacian_v / density
        
        u, v = apply_obstacle(u, v, obstacle)
        
        u_list.append(u.copy())
        v_list.append(v.copy())
    
    return u_list, v_list

u_list, v_list = simulate_fluid_flow_with_obstacle(grid_size, time_steps, viscosity, density, obstacle)

scale=20

fig, ax = plt.subplots()
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
X, Y = np.meshgrid(x, y)
Q = ax.quiver(X, Y, u_list[0], v_list[0], scale=scale, scale_units='xy')

ani = animation.FuncAnimation(fig, update_quiver, fargs=(Q, u_list, v_list), frames=range(time_steps), interval=50, blit=False)
plt.title("Simulated Fluid Flow with Obstacle")
plt.show()

# Save the final frame as an image
final_frame = time_steps - 1
fig, ax = plt.subplots()
Q = ax.quiver(X, Y, u_list[final_frame], v_list[final_frame], scale=scale, scale_units='xy')
plt.title("Final Frame of Simulated Fluid Flow with Obstacle")
plt.savefig('../Images/final_frame.png')
plt.close(fig)
