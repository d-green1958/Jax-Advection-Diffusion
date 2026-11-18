import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax_advection_diffusion.numerics import diffusion
from jax_advection_diffusion.initial_conditions import gaussian

from pytest import approx

def test_diffusion_conservation():
    # create uniform concentration and compute conservation of mass
    c = jnp.full((16,16), 1.0)
    D = 1.0

    assert(diffusion(c, D, 1.0, 1.0).sum() == 0.0)

def test_diffusion_mean():
    # create gaussian distribution and compute mean of diffusion operator
    dx = dy = 0.1
    x,y = jnp.meshgrid(jnp.arange(-1+dx/2, 1, dx), jnp.arange(-1+dy/2, 1, dy), indexing="ij")

    A = 1.0
    sigma = 0.3
    D = 1.0

    c = gaussian(x,y,A,0,0,sigma)
    assert(diffusion(c,D,dx,dy).sum() == approx(0.0))
