import torch
import numpy as np
import os
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(42)

# Define the ExactSolution class
class ExactSolution:
    @staticmethod
    def solution(xy):
        x = xy[:, 0]
        y = xy[:, 1]
        return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

    @staticmethod
    def source_term(xy):
        x = xy[:, 0]
        y = xy[:, 1]
        return -2 * (torch.pi ** 2) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

# Define the PINN class
class PINN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 1)
        )
    
    def forward(self, xy):
        return self.layers(xy)

    @staticmethod
    def compute_loss(model, X_colloc, X_boundary, u_boundary):
        # PDE loss
        X_colloc_pde = X_colloc.clone().detach().requires_grad_(True)
        u = model(X_colloc_pde)
        
        # First derivatives
        du = torch.autograd.grad(u, X_colloc_pde, 
                               grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
        du_dx = du[:, 0]
        du_dy = du[:, 1]
        
        # Second derivatives
        d2u_dx2 = torch.autograd.grad(du_dx, X_colloc_pde, 
                                     grad_outputs=torch.ones_like(du_dx),
                                     create_graph=True)[0][:, 0]
        d2u_dy2 = torch.autograd.grad(du_dy, X_colloc_pde, 
                                     grad_outputs=torch.ones_like(du_dy),
                                     create_graph=True)[0][:, 1]
        
        # Compute PDE residual
        laplacian = d2u_dx2 + d2u_dy2
        f_values = ExactSolution.source_term(X_colloc_pde)
        pde_loss = torch.mean((laplacian - f_values) ** 2)
        
        # Boundary loss
        u_pred = model(X_boundary)
        boundary_loss = torch.mean((u_pred - u_boundary) ** 2)
        
        return pde_loss + boundary_loss

# Generate training data
N = 1000  # Collocation points
N_boundary = 200  # Boundary points

# Collocation points inside the domain
X_colloc = np.random.rand(N, 2)

# Boundary points (x=0, x=1, y=0, y=1)
X_left = np.hstack([np.zeros((N_boundary//4, 1)), np.random.rand(N_boundary//4, 1)])
X_right = np.hstack([np.ones((N_boundary//4, 1)), np.random.rand(N_boundary//4, 1)])
X_bottom = np.hstack([np.random.rand(N_boundary//4, 1), np.zeros((N_boundary//4, 1))])
X_top = np.hstack([np.random.rand(N_boundary//4, 1), np.ones((N_boundary//4, 1))])
X_boundary = np.vstack([X_left, X_right, X_bottom, X_top])

# Convert to tensors
X_colloc_tensor = torch.tensor(X_colloc, dtype=torch.float32)
X_boundary_tensor = torch.tensor(X_boundary, dtype=torch.float32)
u_boundary_tensor = ExactSolution.solution(X_boundary_tensor)  # Should be zero

# Build the model
model = PINN()

# Optimizer and training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = PINN.compute_loss(model, X_colloc_tensor, X_boundary_tensor, u_boundary_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}, Loss: {loss.item():.6f}')

# Evaluate and plot results
x = y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
xy_grid = np.vstack([X.ravel(), Y.ravel()]).T
xy_grid_tensor = torch.tensor(xy_grid, dtype=torch.float32)

with torch.no_grad():
    u_pred = model(xy_grid_tensor).numpy().reshape(X.shape)
    u_exact = ExactSolution.solution(xy_grid_tensor).numpy().reshape(X.shape)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, u_pred, cmap='viridis')
plt.colorbar()
plt.title('Predicted Solution')

plt.subplot(1, 2, 2)
plt.contourf(X, Y, u_exact, cmap='viridis')
plt.colorbar()
plt.title('Exact Solution')

current_file_directory = os.path.dirname(__file__)
images_dir = os.path.join(current_file_directory, 'images')
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'solution_comparison.png'))

error = np.mean((u_pred - u_exact) ** 2)
print(f'Mean Squared Error: {error:.2e}')