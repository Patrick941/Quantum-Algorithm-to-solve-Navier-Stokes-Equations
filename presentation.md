% Quantum Solution to Poissons Equation for Flow Simulation
% Author: Patrick Farmer

# Introduction

- Flow simulation is a critical component in many fields
- The Poisson equation solves for the pressure field in incompressible flow
- Traditional methods can be computationally expensive
- Quantum computing offers a potential speedup for solving PDEs

---

# Background

- Most solutions use Quantum Annealing, by first converting the problem to a QUBO (Quadratic Unconstrained Binary Optimization) problem
- Another common approach for solving PDEs are variational quantum algorithms

<div style="font-size: 0.8em; text-align: center;">
*Ali, Mazen and Kabel, Matthias. "Performance Study of Variational Quantum Algorithms for Solving the Poisson Equation on a Quantum Computer." Physical Review Applied, vol. 20, no. 1, 2023, doi:10.1103/physrevapplied.20.014054.*
</div>

---

# Variational Quantum Imaginary Time Evolution

- Transform the Poisson Equation into a Quantum Hamiltonian
- Prepare a Parameterized Quantum Circuit (Ansatz)
- Simulate Imaginary Time Evolution with VarQITE
- Validate Against Classical Solutions

---

# Quantum Machine Learning

- Dense hidden layers are replaced by rotation gates
- Qubits have "built-in" physical rules while bits have to learn from scratch

---

# Results

---

# Conclusion