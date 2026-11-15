# Jax-Advection-Diffusion
A differential solver for 2D advection-diffusion problems written in Jax-Python.

## Features
The advection-diffusion equation of a scalar $c$ is given by
$$
    \frac{\partial c}{\partial t} + \nabla \cdot \left( \bold{u} c \right) = \nabla \cdot \left( D \nabla c \right),
$$
where $\bold{u} = (u,v)$ is a prescribed velocity field. This is solved using a 
- Cell-centred finite volume method approach
- 1st order upwind advection
- 2nd order central diffusion
- Periodic boundary conditions
- Forward Euler time integration

