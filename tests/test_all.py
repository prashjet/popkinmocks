import pytest
import numpy as np

np.seterr(divide="ignore", invalid="ignore")  # hide some warnings
import popkinmocks as pkm
import copy


def loop_over_dists(cube, stellar_system, light_weighted, density, distribution_list):
    """Helper function to loop over different distributions"""
    for which_dist in distribution_list:
        print(which_dist, light_weighted, density)
        dependent_vars = which_dist.split("_")[0]
        n_dependent = len(dependent_vars)
        is_conditional = "_" in which_dist
        if "x" in dependent_vars:
            n_dependent += 1
        try:
            p = stellar_system.get_p(
                which_dist=which_dist,
                light_weighted=light_weighted,
                density=density,
                collapse_cmps=True,
            )
        except TypeError:
            p = stellar_system.get_p(
                which_dist=which_dist, light_weighted=light_weighted, density=density
            )
        if density:
            dvol = cube.construct_volume_element(dependent_vars)
            p = (p.T * dvol.T).T
        total_prob = np.sum(p, tuple(range(n_dependent)))
        if is_conditional is False:
            assert np.allclose(total_prob, 1.0)
        else:
            # allow for nan in conditionals since denominator can be 0
            assert np.all(np.isclose(total_prob, 1.0) | np.isnan(total_prob))


def test_cube(my_cube):
    """Test sizes of variable edge/value arrays are concordant"""
    cube = my_cube
    for var in ["t", "v", "x1", "x2", "z"]:
        var_edgs = cube.get_variable_edges(var)
        var_cent = cube.get_variable_values(var)
        assert var_edgs.size == var_cent.size + 1


def test_normalisations(my_cube, my_galaxy, distribution_list, my_base_component):
    """Tests that all densities are normalised correctly.

    Loops over all options for evaluating probability functions:
    light-weighting, density, and different component types.

    """
    cube = my_cube
    galaxy = my_galaxy
    base_cmp = my_base_component
    for system in [galaxy, base_cmp]:
        for light_weighted in [False, True]:
            for density in [False, True]:
                loop_over_dists(
                    cube, system, light_weighted, density, distribution_list
                )
    # above miss density evaluations for light-weighted parametric components
    component1 = galaxy.component_list[0]
    loop_over_dists(cube, component1, True, True, distribution_list)


def test_moments(my_cube, my_galaxy, my_base_component):
    """Compare exact moments from mixture model vs discrete approximation

    my_galaxy and my_base_component represent the same denisty but the first is
    a mixture model, the second is represented as a discrete p(t,v,x,z).
    Their moments are calculated in two different ways: using exact formulae
    for mixture models, and a discrete approximation. This test checks that
    the two methods agree withing some tolerance. Only need to check skewness
    and kurtosis because these implicitly use mean and variance.

    """
    cube = my_cube
    galaxy = my_galaxy
    base_cmp = my_base_component
    for lw in [False, True]:
        for dist in ["v_tx", "v_x", "v_t", "v_tz", "v_xz", "v_z"]:  # 'v'
            a = galaxy.get_skewness(dist, light_weighted=lw)
            b = base_cmp.get_skewness(dist, light_weighted=lw)
            error = (a - b) / a
            median_error = np.nanmedian(np.abs(error))
            print(dist, lw, median_error)
            # fairly large (8%) error tolerance since we use a coarse
            # velocity discretisation to allow for tests to run quickly
            assert median_error < 0.08
        for dist in ["t", "x", "z", "t_x", "x_tz", "z_t"]:
            a = galaxy.get_excess_kurtosis(dist, light_weighted=lw)
            b = base_cmp.get_excess_kurtosis(dist, light_weighted=lw)
            error = (a - b) / a
            median_error = np.nanmedian(np.abs(error))
            print(dist, lw, median_error)
            assert median_error < 1e-10


def test_datacube(my_galaxy, my_base_component):
    """Check that datacubes computed using two different methods agree.

    `my_galaxy` is a mixture of parametric components, and uses analytic
    Fourier transforms to evaluate the datacube. `my_base_component`
    is created using a discretised $p(t,v,x,z)$ and uses FFTs to evaluate the
    datacube. Check that the difference per spaxel < 0.05% between two methods.

    """
    galaxy = my_galaxy
    base_cmp = my_base_component
    error = (galaxy.ybar - base_cmp.ybar) / galaxy.ybar
    assert np.median(np.abs(error)) < 0.0005  # i.e. 0.05% error


def test_noise(my_galaxy):
    """Check noise level of ShotNoise is > ConstantSNR for equal SNR

    ShotNoise and ConstantSNR will have equal SNR in the brightest spaxel, but
    SNR drops in fainter spaxels for ShotNoise.

    """
    galaxy = my_galaxy
    shot_noise = pkm.noise.ShotNoise(galaxy)
    yobs_shot_noise = shot_noise.get_noisy_data(snr=100)
    eps_shot_noise = yobs_shot_noise - galaxy.ybar
    mad_shot_noise = np.median(np.abs(eps_shot_noise))
    constant_noise = pkm.noise.ConstantSNR(galaxy)
    yobs_const_snr = constant_noise.get_noisy_data(snr=100)
    eps_const_snr = yobs_const_snr - galaxy.ybar
    mad_const_snr = np.median(np.abs(eps_const_snr))
    assert mad_shot_noise > mad_const_snr


def test_datacube_batch(my_galaxy, my_base_component):
    """Check that datacubes calculated in batches agrees with unbatched version"""
    disk = my_galaxy.component_list[0]
    base_cmp = my_base_component
    for component in [disk, base_cmp]:
        a = copy.copy(component.ybar)  # the unbatched datacube
        for batch_type in ["column", "spaxel"]:
            print(component, batch_type)
            component.evaluate_ybar(batch=batch_type)
            assert np.allclose(a, component.ybar)


def test_from_particle(my_cube):
    """Check that particle histrogramming is as expected

    Compare two histrogrammed versions of p(t,z), evaluated once using pkm
    `get_p('tz', ...)` and once using `np.histogram2d`

    """
    cube = my_cube
    N = 10
    t = np.random.uniform(0, 13, N)  # age [Gyr]
    v = np.random.normal(0, 200.0, N)  # LOS velocity [km/s]
    x1 = np.random.normal(0, 0.4, N)  # x1 position [arbitary]
    x2 = np.random.normal(0, 0.2, N)  # x2 position [arbitary]
    z = np.random.uniform(-2.5, 0.6, N)  # metallicty [M/H]
    simulation = pkm.components.FromParticle(cube, t, v, x1, x2, z)
    p_tz = simulation.get_p("tz", density=True)
    # filter out lost particles
    v_rng = cube.get_variable_extent("v")
    x1_rng = cube.get_variable_extent("x1")
    x2_rng = cube.get_variable_extent("x2")
    idx = np.where(
        (v >= v_rng[0])
        & (v < v_rng[1])
        & (x1 >= x1_rng[0])
        & (x1 < x1_rng[1])
        & (x2 >= x2_rng[0])
        & (x2 < x2_rng[1])
    )
    t_edg = cube.get_variable_edges("t")
    z_edg = cube.get_variable_edges("z")
    p_tz2, _, _ = np.histogram2d(t[idx], z[idx], bins=(t_edg, z_edg), density=True)
    assert np.allclose(p_tz, p_tz2)


def test_plotting(my_cube, my_base_component):
    """Test plotting routines"""
    base_cmp = my_base_component
    cube = my_cube
    p_tz = base_cmp.get_p("tz", density=False)
    p_t = base_cmp.get_p("t", density=False)
    ax_img = cube.imshow(p_tz, view=["t", "z"])
    ax_plt = cube.plot("t", p_t, xspacing="discrete")
    lineplt_ydata = ax_plt.lines[0].get_ydata()
    image2d_ydata = np.sum(ax_img.get_images()[0].get_array(), 0)
    np.allclose(lineplt_ydata, image2d_ydata)


def test_correlation(my_galaxy, n_check=3):
    """Test that correlation coefficients rho satisfty -1 < rho < 1

    Generate three random bivarite marginal or conditional distributions - e.g.
    tv, xz_t, tv_xz - and check that the correlation coefficients

    """
    galaxy = my_galaxy

    def random_distribution_generator():
        all_vars = ["t", "v", "x", "z"]
        dependent_vars = np.random.choice(all_vars, 2, replace=False)
        dependent_vars = np.sort(dependent_vars)
        which_dist = "".join(dependent_vars)  # list of strings to string
        n_conditioners = np.random.choice([0, 1, 2], 1)[0]
        if n_conditioners > 0:
            remaining_vars = np.setdiff1d(all_vars, dependent_vars)
            conditioners = np.random.choice(
                remaining_vars, n_conditioners, replace=False
            )
            conditioners = np.sort(conditioners)
            which_dist = f'{which_dist}_{"".join(conditioners)}'
        return which_dist

    for i in range(n_check):
        which_dist = random_distribution_generator()
        print(i, which_dist)
        correlation = galaxy.get_correlation(which_dist)
        assert np.nanmin(correlation) >= -1.0
        assert np.nanmax(correlation) <= 1.0


def test_l_moments(my_base_component):
    """Test implementation of L-moments

    Checks that L-mean is (almost) equal to mean, and L-skewness and L-kurtosis
    are between -1 and 1.

    """
    cmp = my_base_component
    for lw in [False, True]:
        for dist in ["v_x"]:  # add examples with "v_tz", "x", "t", "t_x", "z_t"
            # check L-mean = mean
            mean = cmp.get_mean(dist, light_weighted=lw)
            l_mean = cmp.get_l_mean(dist, light_weighted=lw)
            error = l_mean/mean - 1.
            median_error = np.nanmedian(np.abs(error))
            print(dist, lw, median_error)
            assert median_error < 0.01
            # check |L-skewness| < 1
            l_skewness = cmp.get_l_skewness(dist, light_weighted=lw)
            assert np.all(np.abs(l_skewness)<=1.)
            # check |L-kurtosis| < 1
            l_kurtosis = cmp.get_l_skewness(dist, light_weighted=lw)
            assert np.all(np.abs(l_kurtosis)<=1.)

# end
