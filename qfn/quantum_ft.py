import numpy as np
import matplotlib.pyplot as plt
import os

class QFT1D:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.N = 2 ** n_qubits
        self.qft_matrix = self._build_qft_matrix()
        self.inv_qft_matrix = self.qft_matrix.conj().T  # Precompute inverse matrix

    def _build_qft_matrix(self):
        """Construct QFT matrix using roots of unity"""
        omega = np.exp(2j * np.pi / self.N)
        return np.array([[omega ** (j * k) for k in range(self.N)] 
                       for j in range(self.N)]) / np.sqrt(self.N)

    def transform(self, state, inverse=False):
        """Apply QFT or inverse QFT to a 1D state"""
        if inverse:
            return self.inv_qft_matrix @ state
        return self.qft_matrix @ state

class QuantumFFT2D:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.size = 2 ** n_qubits
        self.qft_1d = QFT1D(n_qubits)
        
    def forward(self, state_2d):
        """2D QFT through row-wise then column-wise 1D QFT"""
        # Validate input
        assert state_2d.shape == (self.size, self.size)
        
        # Apply QFT to rows
        row_transformed = np.zeros_like(state_2d)
        for i in range(self.size):
            row_transformed[i] = self.qft_1d.transform(state_2d[i])
            
        # Apply QFT to columns
        final_result = np.zeros_like(state_2d)
        for j in range(self.size):
            final_result[:, j] = self.qft_1d.transform(row_transformed[:, j])
            
        return final_result

    def inverse(self, state_2d):
        """Inverse 2D QFT through column-wise then row-wise inverse 1D QFT"""
        # Apply inverse QFT to columns
        col_transformed = np.zeros_like(state_2d)
        for j in range(self.size):
            col_transformed[:, j] = self.qft_1d.transform(state_2d[:, j], inverse=True)
            
        # Apply inverse QFT to rows
        final_result = np.zeros_like(state_2d)
        for i in range(self.size):
            final_result[i] = self.qft_1d.transform(col_transformed[i], inverse=True)
            
        return final_result

if __name__ == "__main__":
    # Configuration (must be power of 2)
    n = 8
    L = 1.0
    x = y = np.linspace(0, L, n, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create normalized Gaussian input
    sigma = 0.1
    u_original = np.exp(-((X-L/2)**2 + (Y-L/2)**2)/(2*sigma**2))
    u_original /= np.sum(u_original)

    # Initialize QFT processor
    n_qubits = int(np.log2(n))
    qfft = QuantumFFT2D(n_qubits)

    # Process transforms
    F = qfft.forward(u_original)
    u_reconstructed = qfft.inverse(F)

    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original image
    cont1 = ax1.contourf(X, Y, u_original, levels=50, cmap='viridis')
    plt.colorbar(cont1, ax=ax1)
    ax1.set_title('Original Image')
    
    # QFT magnitude spectrum
    cont2 = ax2.contourf(X, Y, np.log(np.abs(F) + 1e-10), levels=50, cmap='viridis')
    plt.colorbar(cont2, ax=ax2)
    ax2.set_title('QFT Magnitude (log scale)')
    
    # Reconstructed image
    cont3 = ax3.contourf(X, Y, np.real(u_reconstructed), levels=50, cmap='viridis')
    plt.colorbar(cont3, ax=ax3)
    ax3.set_title('Reconstructed Image')

    # Save results
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/quantum_fft_v2.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Validation metrics
    reconstruction_error = np.max(np.abs(u_original - u_reconstructed))
    print(f"Maximum reconstruction error: {reconstruction_error:.2e}")
    print(f"QFT magnitude range: {np.min(np.abs(F)):.2e} to {np.max(np.abs(F)):.2e}")