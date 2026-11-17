import jax.numpy as jnp


def advection(c, u, v, dx, dy):
    """
    Computes 2D advective fluxes

    Args:
        c : 2D array of concentration values at cell centres
        u : 2D array of x-velocities at cell faces
        v : 2D array of y-velocities at cell faces
        dx : size of x faces
        dy : size of y faces

    Returns:
        2D advective fluxes
    """

    # concentration from left and downwards cells
    c_left = jnp.roll(c, 1, axis=0)
    c_down = jnp.roll(c, 1, axis=1)

    # compute flux in x and y directions
    f_x = 0.5 * ((c + c_left) * u + (c_left - c) * jnp.abs(u))
    f_y = 0.5 * ((c + c_down) * v + (c_down - c) * jnp.abs(v))

    # sum fluxes
    div_f = (
        (jnp.roll(f_x, -1, axis=0) - f_x) / dx
        + (jnp.roll(f_y, -1, axis=1) - f_y) / dy
    )

    return div_f


def diffusion(c, D, dx, dy):
    """
    Computes 2D diffusion of c with diffusivity D

    Args:
        c : 2D array of concentration values at cell centres
        D : Diffusivity constant
        dx : size of x faces
        dy : size of y faces

    Returns:
        2D Diffusive fluxes
    """

    c_left = jnp.roll(c, 1, axis=0)
    c_right = jnp.roll(c, -1, axis=0)

    c_down = jnp.roll(c, 1, axis=1)
    c_up = jnp.roll(c, -1, axis=1)

    return D * (c_left - 2*c + c_right) / dx**2 + D * (c_down - 2*c + c_up) / dy**2