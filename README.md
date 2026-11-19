# Jax Scalar Advection Diffusion Solver
A differential solver for 2D advection-diffusion problems written in Jax-Python.

[![CI (pytest)](https://github.com/d-green1958/Jax-Advection-Diffusion/actions/workflows/python-app.yml/badge.svg?branch=dev&event=push)](https://github.com/d-green1958/Jax-Advection-Diffusion/actions/workflows/python-app.yml)

## Background
This package solves the advection-diffusion equation of a scalar $c$,
```math
\frac{\partial c}{\partial t} + \nabla \cdot \left( \mathbf{u} c \right) = \nabla \cdot \left( D \nabla c \right)
```
where $\mathbf{u} = (u,v)$ is a prescribed velocity field. The solver is implemented in jax using:
- Cell-centred finite volume method approach
- 1st order upwind advection
- 2nd order central diffusion
- Periodic boundary conditions
- Forward Euler time integration
