import jax.numpy as jnp


def gaussian(x, y, amplitude, x0, y0, sigma):
    """Return a 2D Gaussian concentration field.

    Args:
        x: 2D array of cell-centre x coordinates.
        y: 2D array of cell-centre x coordinates.
        amplitude: Peak concentration.
        x0: Gaussian centre x coordinate.
        y0: Gaussian centre y coordinate.
        sigma: Standard deviation of the Gaussian.

    Returns:
        2D array containing the Gaussian concentration field.
    """
    return amplitude * jnp.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
