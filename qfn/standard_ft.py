from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

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