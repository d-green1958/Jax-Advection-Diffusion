import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax_advection_diffusion.numerics import advection

def test_advection_conservation():
    # create uniform concentration  and compute conservation of mass
    c = jnp.full((16,16), 1.0)
    u = jnp.ones_like(c)
    v = jnp.ones_like(c)

    assert(advection(c, u, v, 1.0, 1.0).sum() == 0.0)