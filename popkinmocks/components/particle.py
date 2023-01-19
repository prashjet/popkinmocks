import numpy as np
from . import base

class FromParticle(base.Component):
    """A component from star particle data from simulations

    This is a wrapper around `base.Component` which makes a 5D histogram of the
    particle data in (t, v, x, z) space. Particles should be mass-weighted i.e.
    each particle should represent the same stellar mass. If this in not the 
    case for your particle, you can provide the `weights` argument which should
    re-weight particles so that they are mass-weighted. All particle data 
    should be provided as 1D arrays of the same length and same particle order.

    Args:
        cube: a pkm.mock_cube.mockCube.
        t (array): ages of star particles (Gyr)
        v (array): line-of-sight velocities of star particles (km/s)
        x1 (array): x1 position of star particles (units consistent with cube)
        x2 (array): x2 position of star particles (units consistent with cube)
        z (array): metallicty of star particles ([M/H])
        weights (array): optional weights to ensure particles are mass-weighted

    """
    def __init__(self, cube, t, v, x1, x2, z, weights=None):
        particle_data = [t, v, x1, x2, z]
        varlist = ['t', 'v', 'x1', 'x2', 'z']
        edges = [cube.get_variable_edges(var) for var in varlist]
        n_total = len(t)
        n_in_bounds = np.sum(
            (t >= edges[0][0]) & (t <= edges[0][-1]) &
            (v >= edges[1][0]) & (v <= edges[1][-1]) &
            (x1 >= edges[2][0]) & (x1 <= edges[2][-1]) &
            (x2 >= edges[3][0]) & (x2 <= edges[3][-1]) &
            (z >= edges[4][0]) & (z <= edges[4][-1])
        )
        n_lost = n_total - n_in_bounds
        warning_string = f"Note: {n_lost}/{n_total} particles out of bounds. "
        warning_string += "N lost per variable:\n"
        for pdat, edg, var in zip(particle_data, edges, varlist):
            lo, hi = edg[0], edg[-1]
            n_lost_per_var = np.sum((pdat < lo) | (pdat>hi))
            warning_string += f"\t{var} : {n_lost_per_var}"
            if var != 'z': warning_string += '\n'
        print(warning_string)
        p_tvxz, _ = np.histogramdd(
            [t, v, x1, x2, z], 
            bins=edges, 
            density=True,
            weights=weights)
        log_p_tvxz = np.log(p_tvxz)
        super().__init__(cube=cube, log_p_tvxz=log_p_tvxz)
