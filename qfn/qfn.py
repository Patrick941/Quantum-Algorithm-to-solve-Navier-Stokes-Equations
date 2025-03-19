from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import os

class FourierTransform(ABC):
    @abstractmethod
    def forward(self, f):
        """Compute the forward Fourier transform of the 2D array f."""
        pass
    
    @abstractmethod
    def inverse(self, F):
        """Compute the inverse Fourier transform of the 2D array F."""
        pass

class FFT2D(FourierTransform):
    def forward(self, f):
        return np.fft.fft2(f)
    
    def inverse(self, F):
        return np.fft.ifft2(F).real

class PoissonSolver:
    def __init__(self, fourier_transform):
        self.fourier = fourier_transform
    
    def solve(self, f):
        F = self.fourier.forward(f)
        nx, ny = f.shape
        
        # Corrected wavenumbers with grid spacing scaling
        kx = 2 * np.pi * np.fft.fftfreq(nx) * nx  # Account for physical grid spacing
        ky = 2 * np.pi * np.fft.fftfreq(ny) * ny  # Account for physical grid spacing
        
        Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
        K_sq = Kx**2 + Ky**2
        
        # Handle zero wavenumber (DC component)
        K_sq[K_sq == 0] = np.inf
        U = -F / K_sq
        
        # Set DC component to zero (solution uniqueness)
        U[0, 0] = 0
        
        u = self.fourier.inverse(U)
        return u

# Example usage and test
if __name__ == "__main__":
    # Parameters
    nx, ny = 64, 64
    Lx, Ly = 1.0, 1.0  # Physical domain size
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Exact solution and source term
    u_exact = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    f = -((2 * np.pi)**2 + (2 * np.pi)**2) * u_exact  # -∇²u_exact
    
    # Solve using corrected Poisson solver
    fft_transformer = FFT2D()
    solver = PoissonSolver(fft_transformer)
    u_computed = solver.solve(f)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Numerical solution plot
    plt.subplot(1, 2, 1)
    cont = plt.contourf(X, Y, u_computed, levels=50, cmap='viridis')
    plt.colorbar(cont)
    plt.title('Fourier Method Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Exact solution plot
    plt.subplot(1, 2, 2)
    cont_exact = plt.contourf(X, Y, u_exact, levels=50, cmap='viridis')
    plt.colorbar(cont_exact)
    plt.title('Exact Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Save figure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    plt.savefig(os.path.join(images_dir, 'poisson_solution.png'), bbox_inches='tight')
    plt.close()
    
    # Error analysis
    abs_error = np.abs(u_computed - u_exact)
    max_error = np.max(abs_error)
    mse = np.mean(abs_error**2)
    
    print(f"Maximum absolute error: {max_error:.2e}")
    print(f"Mean squared error: {mse:.2e}")

    # Additional error visualization
    plt.figure(figsize=(6, 5))
    cont_error = plt.contourf(X, Y, abs_error, levels=50, cmap='hot')
    plt.colorbar(cont_error)
    plt.title('Absolute Error')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(os.path.join(images_dir, 'error_distribution.png'), bbox_inches='tight')
    plt.close()