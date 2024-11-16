# A quantum annealing approach to fluid dynamics problems solving Navier-Stokes equations

"Offering a potential exponential speedup over classical methods"
Applications in "weather forecasting, aircraft design and determining the magento-hydrodynamics of plasmas in space and in nuclear fision."

"Little work has been done on finding quantum algorithms for the Navier-Stokes equations. The first known works is based on a quantum lattice-gas model"

"The current algorithms are able to solve at least part of the Navier-Stokes equations"

"The approach used in this study is based on spectral methods, which avoid some potential issues like the need of repeated communication a classical and quantum computer"

$$
\begin{align}
\partial_t \rho + \nabla \cdot (\rho \vec{v}) &= 0 \quad (1) \\
\partial_t (\rho \vec{v}) + \nabla \cdot (\rho v^2 + P - \tau) &= 0 \quad (2) \\
\partial_t \left[ \rho \left( e + \frac{v^2}{2} \right) \right] + \nabla \cdot \left[ \rho \left( e + \frac{v^2}{2} \right) \vec{v} + P \vec{v} - \tau \vec{v} + \kappa \nabla T \right] &= 0
\end{align}
$$

A goal will be to achieve similar error to classical methods.

# Towards solving the Navier-Stokes equation on quantum computers
Current classical methods to describe turbulence include Direct Numerical Simulation (DNS), Large Eddy Simulation (LES) and Reynolds-Averaged Navier-Stokes (RANS).

"Our focus is to study how a fluid dynamical system, starting with simple transient channel flows can be transformed to a form suitable for D-Wave quantum annealers."

The steps for this solution were to first transofmr the problem to a binary fixed point arithmetic problem and then pose this problem as a least squares problem to convert it to a quadratic unconstrained binary optimization (QUBO) problem.

# Quantum Computing of Fluid Dynamics Using the Hydrodynamic Schrödinger Equation
"The NSE is notoriously difficult to be full simulated on a classical computer at a large Reynolds number."