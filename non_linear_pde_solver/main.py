import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def simulate_fluid_flow(grid_size, time_steps, viscosity, density):
    u = np.ones((grid_size, grid_size))  # Initial condition: uniform flow from left to right
    v = np.zeros((grid_size, grid_size))
    
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

def update_quiver(num, Q, u_list, v_list, obstacle_img):
    Q.set_UVC(u_list[num], v_list[num])
    obstacle_img.set_data(obstacle)
    return Q, obstacle_img

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
    u = np.ones((grid_size, grid_size))  # Initial condition: uniform flow from left to right
    v = np.zeros((grid_size, grid_size))
    
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

fig, ax = plt.subplots()
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
X, Y = np.meshgrid(x, y)
Q = ax.quiver(X, Y, u_list[0], v_list[0])

# Display the obstacle
obstacle_img = ax.imshow(obstacle, extent=[0, 1, 0, 1], origin='lower', cmap='gray', alpha=0.5)

ani = animation.FuncAnimation(fig, update_quiver, fargs=(Q, u_list, v_list, obstacle_img), frames=range(time_steps), interval=50, blit=False)
plt.title("Simulated Fluid Flow with Obstacle")
plt.show()
