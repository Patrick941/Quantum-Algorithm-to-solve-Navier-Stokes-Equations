import numpy as np
import matplotlib.pyplot as plt
import os

class FFT2D:
    def forward(self, f):
        """
        Compute 2D FFT by:
        1. Applying 1D FFT to each row
        2. Applying 1D FFT to each column of the result
        """
        # Convert to complex type to handle intermediate results
        f_complex = f.astype(np.complex128)
        
        # FFT over rows (axis=1)
        for i in range(f_complex.shape[0]):
            f_complex[i] = np.fft.fft(f_complex[i])
        
        # FFT over columns (axis=0)
        for j in range(f_complex.shape[1]):
            f_complex[:, j] = np.fft.fft(f_complex[:, j])
            
        return f_complex
    
    def inverse(self, F):
        """
        Compute 2D inverse FFT by:
        1. Applying 1D IFFT to each column
        2. Applying 1D IFFT to each row of the result
        """
        # Make copy to preserve original data
        F_copy = F.copy()
        
        # IFFT over columns (axis=0)
        for j in range(F_copy.shape[1]):
            F_copy[:, j] = np.fft.ifft(F_copy[:, j])
        
        # IFFT over rows (axis=1)
        for i in range(F_copy.shape[0]):
            F_copy[i] = np.fft.ifft(F_copy[i])
            
        return F_copy.real

# Example usage and test
if __name__ == "__main__":
    # Parameters
    nx, ny = 64, 64
    Lx, Ly = 1.0, 1.0
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create test input (2D Gaussian)
    sigma = 0.1
    u_original = np.exp(-((X - Lx/2)**2 + (Y - Ly/2)**2) / (2*sigma**2))
    
    # Instantiate and process
    fft_processor = FFT2D()
    F = fft_processor.forward(u_original)
    u_reconstructed = fft_processor.inverse(F)
    
    # Verification
    print("Reconstruction accurate?", np.allclose(u_original, u_reconstructed, atol=1e-10))
    
    # Visualization
    plt.figure(figsize=(18, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    cont = plt.contourf(X, Y, u_original, levels=100, cmap='viridis')
    plt.colorbar(cont)
    plt.title('Original Image')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # FFT magnitude spectrum
    plt.subplot(1, 3, 2)
    F_magnitude = np.log(np.abs(F) + 1e-10)
    cont_fft = plt.contourf(X, Y, F_magnitude, levels=100, cmap='viridis')
    plt.colorbar(cont_fft)
    plt.title('FFT Magnitude (log scale)')
    plt.xlabel('kx')
    plt.ylabel('ky')
    
    # Reconstructed image
    plt.subplot(1, 3, 3)
    cont_rec = plt.contourf(X, Y, u_reconstructed, levels=100, cmap='viridis')
    plt.colorbar(cont_rec)
    plt.title('Reconstructed Image')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Save figure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    plt.savefig(os.path.join(images_dir, 'standard_fourier.png'), dpi=300, bbox_inches='tight')
    plt.close()