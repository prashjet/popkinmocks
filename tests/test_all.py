import pytest
import numpy as np
import popkinmocks as pkm

def loop_over_dists(cube,
                    stellar_system,
                    light_weighted,
                    density,
                    distribution_list):
    # loop over distributions
    for which_dist in distribution_list:
        print(which_dist, light_weighted, density)
        dependent_vars = which_dist.split('_')[0]
        n_dependent = len(dependent_vars)
        is_conditional = '_' in which_dist
        if 'x' in dependent_vars: n_dependent += 1
        try:
            p = stellar_system.get_p(
                which_dist=which_dist,
                light_weighted=light_weighted,
                density=density,
                collapse_cmps=True)
        except TypeError:
            p = stellar_system.get_p(
                which_dist=which_dist,
                light_weighted=light_weighted,
                density=density)
        if density:
            dvol = cube.construct_volume_element(dependent_vars)
            p = (p.T * dvol.T).T
        total_prob = np.sum(p, tuple(range(n_dependent)))
        if is_conditional is False:
            assert np.allclose(total_prob, 1.)
        else:
            # allow for nan in conditionals since denominator can be 0
            assert np.all(np.isclose(total_prob, 1.) | np.isnan(total_prob))

def test_normalisations(my_cube,
                        my_galaxy,
                        distribution_list,
                        my_base_component):
    """Tests the normalisations of all densities evaluated for a component

    """
    cube = my_cube
    galaxy = my_galaxy
    base_cmp = my_base_component
    for system in [galaxy, base_cmp]:
        for light_weighted in [False, True]:
            for density in [False, True]:
                loop_over_dists(
                    cube,
                    system,
                    light_weighted,
                    density,
                    distribution_list)
    # above miss density evaluations for light-weighted parametric components
    component1 = galaxy.component_list[0]
    loop_over_dists(cube, component1, True, True, distribution_list)

def test_moments(my_cube,
                 my_galaxy,
                 my_base_component):
    """Tests moments exact vs. numerical agree

    """
    cube = my_cube
    galaxy = my_galaxy
    base_cmp = my_base_component
    for lw in [False, True]:
        for dist in ['v_tx', 'v_x', 'v_t', 'v_tz', 'v_xz', 'v_z', 'v']:
            a = galaxy.get_skewness(dist, light_weighted=lw)
            b = base_cmp.get_skewness(dist, light_weighted=lw)
            error = (a-b)/a
            median_error = np.nanmedian(np.abs(error))
            print(dist, lw, median_error)
            # fairly large (6%) error tolerance since velocity discretisation
            # is coarse (20 km/s) to allow for tests to run quickly
            assert median_error<0.06
        for dist in ['t', 'x', 'z', 't_x', 'x_tz', 'z_t']:
            a = galaxy.get_kurtosis(dist, light_weighted=lw)
            b = base_cmp.get_kurtosis(dist, light_weighted=lw)
            error = (a-b)/a
            median_error = np.nanmedian(np.abs(error))
            print(dist, lw, median_error)
            assert median_error<1e-10

def test_datacube(my_galaxy, my_base_component):
    """Check that datacubes computed using two different methods agree.
    
    `my_galaxy` is a mixture of parametric components, and uses analytic 
    Fourier transforms to evaluate the datacube. `my_base_component` 
    is created using a discretised $p(t,v,x,z)$ and uses FFTs to evaluate the 
    datacube. Check that the difference per spaxel < 0.05% between two methods.

    """
    galaxy = my_galaxy
    base_cmp = my_base_component
    error = (galaxy.ybar - base_cmp.ybar)/galaxy.ybar
    assert np.median(np.abs(error)) < 0.0005 # i.e. 0.05% error


# end
