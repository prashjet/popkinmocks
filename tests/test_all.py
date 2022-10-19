import pytest
import numpy as np
import popkinmocks as pkm

def loop_over_dists(stellar_system,
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
        p = stellar_system.get_p(
            which_dist=which_dist,
            light_weighted=light_weighted,
            density=density,
            collapse_cmps=True)
        if density:
            dvol = stellar_system.construct_volume_element(dependent_vars)
            p = (p.T * dvol.T).T
        total_prob = np.sum(p, tuple(range(n_dependent)))
        if is_conditional is False:
            assert np.allclose(total_prob, 1.)
        else:
            # allow for nan in conditionals since denominator can be 0
            assert np.all(np.isclose(total_prob, 1.) | np.isnan(total_prob))

def test_normalisations(my_three_component_cube, distribution_list):
    """Tests the normalisations of all densities evaluated for a component

    """
    cube = my_three_component_cube
    for light_weighted in [False, True]:
        for density in [False, True]:
            loop_over_dists(cube, light_weighted, density, distribution_list)





# end
