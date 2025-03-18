from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

# Secret key and encryption setup
secret_key = '11'
plaintext = '01'
ciphertext = '10'  # plaintext XOR secret_key = '01' ⊕ '11' = '10'

# Create quantum circuit
qc = QuantumCircuit(2, 2)

# Step 1: Superposition
qc.h([0, 1])

# Step 2: Oracle for key '11' (marks |11> with a phase flip)
qc.cz(0, 1)

# Step 3: Grover's diffusion operator
qc.h([0, 1])
qc.x([0, 1])
qc.cz(0, 1)
qc.x([0, 1])
qc.h([0, 1])

# Measure
qc.measure([0, 1], [0, 1])

# Simulate
simulator = Aer.get_backend('qasm_simulator')
result = execute(qc, simulator, shots=1000).result()
counts = result.get_counts()

# Verify the key
print("Measurement results:", counts)
print("\nVerification:")
if secret_key in counts:
    print(f"Key '{secret_key}' found successfully!")
    # Decrypt ciphertext with found key
    decrypted = ''.join(str(int(c) ^ int(k)) for c, k in zip(ciphertext, secret_key))
    print(f"Decrypted ciphertext '{ciphertext}' with key '{secret_key}': {decrypted}")
    assert decrypted == plaintext, "Decryption failed!"
else:
    print("Key not found. Algorithm failed.")

plot_histogram(counts)