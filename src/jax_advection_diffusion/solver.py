import jax

from jax_advection_diffusion.numerics import rhs


def step(c, u, v, D, dx, dy, dt):
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

    Returns:
        c : Evolved concentration field 
    """

    return  c + dt*rhs(c,u,v,D,dx,dy)    

def simulate(c, u, v, D, dx, dy, dt, N):
    """
    Advances concentration field by N timesteps

    Args:
        c : Transported concentration
        u : Horizontal velocity field
        v : Vertical velocity field
        D : Diffusivity
        D : Diffusivity constant
        dx : size of x faces
        dy : size of y faces
        dt : timestep
        N : number of timesteps

    Returns:
        Final concentration field
    """

    def scan_func(c, _):
        c_next = step(c, u, v, D, dx, dy, dt)
        return c_next, None

    c_final = jax.lax.scan(scan_func, c, None, length=N)

    return c_final