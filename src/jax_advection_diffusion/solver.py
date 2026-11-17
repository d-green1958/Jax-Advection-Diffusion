from jax_advection_diffusion.numerics import advection, diffusion
import jax.numpy as jnp

def forward_euler(c, u, v, D, dx, dy, dt, t_end, t_start=0.0):
    """
    Evolves c in time using 1st order forward euler scheme

    Args:
        c : Transported concentration
        u : Horizontal velocity field
        v : Vertical velocity field
        D : Diffusivity constant
        dx : size of x faces
        dy : size of y faces
        dt : timestep
        t_end : End time
        t_start (optional): Start time. Defaults to 0.

    Returns:
        t : Solution time
        c : Concentration field 
    """
    t = t_start
    while (t < t_end):
        c = c + dt*diffusion(c, D, dx, dy) - dt*advection(c, u, v, dx, dy)        
        t += dt
    return t, c


