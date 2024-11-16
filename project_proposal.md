---
header-includes:
    - \usepackage{titling}
    - \usepackage{titlesec}
    - \titleformat{\section}{\centering\Large\bfseries}{\thesection}{1em}{}
    - \usepackage[margin=1in]{geometry}
---

 \
 \
 \

# Quantum Algorithms
 \

# **Solving a Non-Linear Partial Differential Equation**
 \


# Patrick Farmer 20331828
# 16-11-2024
 \

![](https://www.tcd.ie/media/tcd/site-assets/images/tcd-logo.png)

\clearpage

# Introduction

There have already been a lot of papers and research that has gone into the solving of linear partial differential equations (PDEs) using quantum computer for an exponential speedup. In the case of linear PDEs quantum algorithms are very well suited and reasonably simple to implement. However, there is somme added complexity for non-linear PDEs. The Navier-Stokes equations are a set of non-linear PDEs that describe the motion of fluid. "They are used in many fields such as weather forecasting, aircraft design and determining the magneto-hydrodynamics of plasmas in space and in nuclear fission" (2). "The Navier-Stokes equations are notoriously difficult to solve on classical computers at large Reynolds numbers" (3) and so a quantum algorithm could offer a significant speedup. 

# Problem Definition

The Navier-Stokes equations are a set of non-linear PDEs. The equations are given by:

$$
\nabla u = 0
$$

$$
\rho \frac{du}{dt} = -\nabla p + \mu \nabla^2 u + F
$$

A common use case for the Navier-Stokes equations is to model air flow around an aircraft. The quantum algorithm developed in this paper will be used to solve the Navier-Stokes and simulate the air flow around an aircraft. The simulation will be very simple and will only consider a 2D flow around some object. The same simulation will be run on a classical computer to have a visual comparison of the results in addition to the numerical comparison. 

# Objectives

# Conclusion

# Bibliography
1. Nadiga, B., & Karra, S. (2024). Towards Solving the Navier-Stokes Equation on Quantum Computers. eScholarship, University of California. Available at: https://arxiv.org/abs/1904.09033
2. IEEE. (2024). A Quantum Annealing Approach to Fluid Dynamics Problems Solving Navier-Stokes Equations. IEEE Xplore. Available at: https://ieeexplore.ieee.org/document/10612316
3. ArXiv. (2024). Quantum Computing of Fluid Dynamics Using the Hydrodynamic Schrödinger Equation. Available at: https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.033182