import torch
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# Configure PyTorch defaults
torch.set_default_dtype(torch.float32)
torch.manual_seed(42)

# Enhanced quantum circuit design
n_qubits = 4  # Number of qubits
dev = qml.device("default.qubit", wires=n_qubits)

def quantum_circuit(params, x):
    """
    Enhanced quantum circuit with data encoding and parametrized layers.
    """
    # Data encoding
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        qml.RY(x[0] * np.pi, wires=i)  # Encode x-coordinate
        qml.RZ(x[1] * np.pi, wires=i)  # Encode y-coordinate
    
    # Parametrized layers
    for layer in params:
        for i in range(n_qubits):
            qml.RX(layer[i], wires=i)
            qml.RY(layer[i + n_qubits], wires=i)
        
        # Enhanced entanglement
        for i in range(n_qubits - 1):
            qml.CZ(wires=[i, i + 1])
        qml.CZ(wires=[n_qubits - 1, 0])  # Cyclic entanglement
    
    # Multi-qubit measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Create QNode with explicit dtype control
qnode = qml.QNode(quantum_circuit, dev, interface="torch")

class EnhancedQuantumPINN(torch.nn.Module):
    """
    Hybrid quantum-classical model with enhanced architecture.
    """
    def __init__(self, n_layers=5):
        super().__init__()
        self.n_layers = n_layers
        
        # Classical preprocessor
        self.classical = torch.nn.Sequential(
            torch.nn.Linear(2, 16),  # Input (x, y) -> 16 features
            torch.nn.Tanh(),
            torch.nn.Linear(16, 2 * n_qubits * n_layers),  # Output quantum parameters
            torch.nn.Tanh()
        )
        
        # Quantum circuit
        self.quantum = qnode
        
        # Post-processor
        self.post_process = torch.nn.Sequential(
            torch.nn.Linear(n_qubits, 8),  # Quantum outputs -> 8 features
            torch.nn.Tanh(),
            torch.nn.Linear(8, 1)  # Final output (solution u)
        )

    def forward(self, xy):
        """
        Forward pass through the hybrid model.
        """
        # Input processing
        xy = xy.float()
        params = self.classical(xy).reshape(-1, self.n_layers, 2 * n_qubits)
        
        # Quantum computations
        quantum_outs = []
        for point in range(xy.shape[0]):
            out = self.quantum(params[point], xy[point])
            quantum_outs.append(torch.stack(out).float())  # Ensure float32
        
        quantum_outs = torch.stack(quantum_outs)
        return self.post_process(quantum_outs)

# Define PDE and boundary conditions
def f(xy):
    """
    Source term for the Poisson equation.
    """
    x = xy[:, 0]
    y = xy[:, 1]
    return -2 * (torch.pi ** 2) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

def exact_solution(xy):
    """
    Exact solution for validation.
    """
    x = xy[:, 0]
    y = xy[:, 1]
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

# Generate training data
N = 1000  # Collocation points
N_boundary = 200  # Boundary points

# Collocation points inside the domain
X_colloc = torch.rand(N, 2, dtype=torch.float32, requires_grad=True)

# Boundary points (x=0, x=1, y=0, y=1)
X_left = torch.cat([torch.zeros(N_boundary // 4, 1), torch.rand(N_boundary // 4, 1)], dim=1)
X_right = torch.cat([torch.ones(N_boundary // 4, 1), torch.rand(N_boundary // 4, 1)], dim=1)
X_bottom = torch.cat([torch.rand(N_boundary // 4, 1), torch.zeros(N_boundary // 4, 1)], dim=1)
X_top = torch.cat([torch.rand(N_boundary // 4, 1), torch.ones(N_boundary // 4, 1)], dim=1)
X_boundary = torch.cat([X_left, X_right, X_bottom, X_top], dim=0).float()

# Physics-informed loss function
def physics_loss(model, X_colloc, X_boundary):
    """
    Loss function combining PDE residual and boundary conditions.
    """
    # PDE Loss
    u = model(X_colloc)
    du = torch.autograd.grad(u, X_colloc, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx, du_dy = du[:, 0], du[:, 1]
    
    d2u_dx2 = torch.autograd.grad(du_dx, X_colloc, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0][:, 0]
    d2u_dy2 = torch.autograd.grad(du_dy, X_colloc, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0][:, 1]
    
    laplacian = d2u_dx2 + d2u_dy2
    pde_loss = torch.mean((laplacian - f(X_colloc)) ** 2)
    
    # Boundary Loss (weighted)
    u_boundary = model(X_boundary)
    boundary_loss = 10.0 * torch.mean(u_boundary ** 2)  # Weighted boundary term
    
    return pde_loss + boundary_loss

# Initialize model
model = EnhancedQuantumPINN(n_layers=5)

# Optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

# Training loop with early stopping
epochs = 1000
best_loss = float('inf')
patience = 50
no_improve = 0
good_enough = 1

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = physics_loss(model, X_colloc, X_boundary)
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    scheduler.step(loss)
    
    # Early stopping
    if loss < best_loss:
        best_loss = loss
        no_improve = 0
    else:
        no_improve += 1
        
    if loss < good_enough:
        print(f"Converged at epoch {epoch}")
        break
        
    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
    
    print(f"Epoch {epoch:4d}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

# Visualization
x = y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
xy_grid = torch.tensor(np.vstack([X.ravel(), Y.ravel()]).T, dtype=torch.float32)

with torch.no_grad():
    u_pred = model(xy_grid).numpy().reshape(X.shape)
    u_exact = exact_solution(xy_grid).numpy().reshape(X.shape)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, u_pred, cmap='viridis')
plt.colorbar()
plt.title('Quantum PINN Solution')

plt.subplot(1, 2, 2)
plt.contourf(X, Y, u_exact, cmap='viridis')
plt.colorbar()
plt.title('Exact Solution')

plt.show()

# Compute mean squared error
mse = np.mean((u_pred - u_exact) ** 2)
print(f"Mean Squared Error: {mse:.4f}")