from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter

from qiskit import Aer
backend = Aer.get_backend('statevector_simulator')



import numpy as np
from scipy import sparse
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 16})  # enlarge matplotlib fonts

from qiskit.opflow import MatrixOp



total_res = {}

N = 4   # number of qubits
dim = 2**N # dimension of the operator A

# Setup a tridiagonal matrix
k = [np.ones(dim-1), -2*np.ones(dim), np.ones(dim-1)]
offset = [-1, 0, 1]
A = diags(k,offset).toarray()

# Setup the driving term f(x) = x
b = np.linspace(0, 1, dim)
h = 1/(dim-1)
sampled_b = b*(h**2)
bt = np.linspace(0, 1, dim)

# Setup the Dirichlet B.C.s
phi_a, phi_b = 0, 0
sampled_b[0] -= phi_a
sampled_b[dim-1] -= phi_b
norm = np.linalg.norm(sampled_b)
sampled_b = sampled_b/norm

# Solve the linear system of equations
x = np.linalg.solve(A, sampled_b)
f = np.linalg.norm(x)
x = x/f

# Build Hamiltonian
sampled_b = sampled_b.reshape([dim, 1])
Hamiltonian = A@(np.eye(dim)- sampled_b@sampled_b.T)@A
# print(Hamiltonian)

print("Classical solution:\n", x)
eig_val, eig_state = np.linalg.eig(Hamiltonian)
# print("Eigenvalues:\n", eig_val)
# print(min(eig_val))
vec = eig_state[:,-1]
# print(eig_state)
print("Eigenvector:\n", -vec)

# Transform into Pauli operators
H_op = MatrixOp(Hamiltonian).to_pauli_op()
print("Lenth of Pauli String:",len(H_op))
# print(H_op)



from qiskit.circuit.library import EfficientSU2

depth = 5 # depth of ansatz
ansatz = EfficientSU2(N, entanglement='linear', reps=depth, skip_final_rotation_layer=True).decompose()
ansatz.draw(fold=300)



ansatz_opt = transpile(ansatz, backend=backend, optimization_level=3)

print('number and type of gates in the cirucit:', ansatz_opt.count_ops())
print('number of parameters in the circuit:', ansatz_opt.num_parameters)
ansatz_opt.draw(fold=300)

from qiskit.algorithms.optimizers import SPSA, COBYLA, L_BFGS_B, NELDER_MEAD, SLSQP, ADAM, AQGD, CG, POWELL, QNSPSA

# optimizer = SPSA(maxiter=500)
optimizer  = L_BFGS_B(maxiter=5000)
# optimizer  = ADAM(maxiter=200, lr=0.2)
# optimizer  = AQGD(maxiter=1000, eta=1.0, tol=1e-06, momentum=0.25, param_tol=1e-06, averaging=10)
# optimizer  = POWELL()
# optimizer  = COBYLA(maxiter=10000)
# optimizer  = SLSQP(maxiter=10000)



from qiskit.utils import QuantumInstance
from qiskit.algorithms import VQE

quantum_instance = QuantumInstance(backend=backend, seed_simulator=28, seed_transpiler=28,
                                        basis_gates=None,
                                        optimization_level=3)




best_result = 99999

vqe = VQE(ansatz_opt, optimizer,quantum_instance=quantum_instance,initial_point=2*np.pi*np.random.rand(ansatz_opt.num_parameters))
result = vqe.compute_minimum_eigenvalue(H_op)
quantum_solution = -1*np.abs(result.eigenstate).real
print(quantum_solution)

if result.eigenvalue.real < best_result:
    best_result = result.eigenvalue.real
    kept_result = result
print("Current round using ansatz TwoLocal with depth {}, found eigenvalue {}. Best so far {}".format(depth, result.eigenvalue.real,best_result))
total_res.update({(N, depth):kept_result})

t = np.arange(0., 1., 0.02)
res = (t**3-t)/6
norm_res = np.linalg.norm(res)
res_norm = res/norm_res

xt = np.arange(0,1,1/dim)
exact = [1/6*(x**3-x) for x in np.arange(0,1,1/dim)]
norm = np.linalg.norm(exact)
exact = exact/norm

# red dashes, blue squares and green triangles
plt.plot(t, res_norm, 'r-', label='analytical')
plt.plot(bt, x, 'o-.', label='classical')
plt.plot(bt, quantum_solution, 'gx--', label='quantum')
# plt.legend()
plt.legend(loc="lower left")
plt.xlabel('Boundary [0,1]')
plt.ylabel('Solution Profile')
plt.title("4-qubit VQE for Poisson Eqn, TwoLocal, BFGS")
plt.grid(linestyle = '--', linewidth = 0.5)
# plt.show()
# plt.savefig("Poisson.png", bbox_inches='tight', dpi=300)