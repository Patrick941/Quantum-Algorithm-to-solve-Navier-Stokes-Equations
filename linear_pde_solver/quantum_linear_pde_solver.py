# Import necessary modules from Qiskit
from qiskit import Aer
from qiskit_algorithms.factorizers import Shor
from qiskit.utils import QuantumInstance
import numpy as np

# Set up the quantum simulator backend
backend = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024)

# Define a weak RSA modulus (a small composite number)
N = 15

# Initialize Shor's algorithm with the quantum instance
shor = Shor(quantum_instance=quantum_instance)

# Run the algorithm to factorize the number
result = shor.factorize(N)

# Output the factors found by the algorithm
print(f"Factors of {N}: {result.factors}")

# Verify that the product of the factors equals the original number
if np.prod(result.factors[0]) == N:
    print("Verification successful: The product of the factors equals", N)
else:
    print("Verification failed")
