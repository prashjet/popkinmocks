import pytest
import matplotlib.pyplot as plt
import numpy as np
import popkinmocks as pkm

################################
# growing disks
################################

@pytest.fixture(scope="module")
def my_component():
    ssps = pkm.model_grids.milesSSPs()
    ssps.logarithmically_resample(dv=100.)
    ssps.calculate_fourier_transform()
    ssps.get_light_weights()
    cube = pkm.ifu_cube.IFUCube(ssps=ssps, nx=9, ny=10)
    gc1 = pkm.components.growingDisk(cube=cube, rotation=0., center=(0,0))
    gc1.set_p_t(lmd=2., phi=0.8)
    gc1.set_p_x_t(sig_x_lims=(0.5, 0.2),
                  sig_y_lims=(0.03, 0.1),
                  alpha_lims=(1.2, 0.8))
    gc1.set_t_dep(sig_x=0.5,
                  sig_y=0.1,
                  alpha=3.,
                  t_dep_in=0.5,
                  t_dep_out=5.)
    gc1.set_p_z_tx()
    gc1.set_mu_v(sig_x_lims=(0.5, 0.1),
                 sig_y_lims=(0.1, 0.1),
                 rmax_lims=(0.5, 1.1),
                 vmax_lims=(250., 100.),
                 vinf_lims=(60., 40.))
    gc1.set_sig_v(sig_x_lims=(0.7, 0.1),
                  sig_y_lims=(0.2, 0.1),
                  alpha_lims=(3.0, 2.5),
                  sig_v_in_lims=(70., 50.),
                  sig_v_out_lims=(80., 60.))
    gc1.evaluate_ybar()
    return ssps, cube, gc1



@pytest.fixture(scope="module")
def my_second_component(my_component):
    ssps, cube, gc1 = my_component
    gc2 = pkm.components.growingDisk(cube=cube,
                                     rotation=10.,
                                     center=(0.05,-0.07))
    gc2.set_p_t(lmd=1.6, phi=0.3)
    gc2.set_p_x_t(sig_x_lims=(0.9, 0.5),
                  sig_y_lims=(0.9, 0.5),
                  alpha_lims=(1.1, 1.1))
    gc2.set_t_dep(sig_x=0.9,
                  sig_y=0.8,
                  alpha=1.1,
                  t_dep_in=6.,
                  t_dep_out=1.)
    gc2.set_p_z_tx()
    gc2.set_mu_v(sig_x_lims=(0.5, 0.4),
                 sig_y_lims=(0.4, 0.6),
                 rmax_lims=(0.7, 0.2),
                 vmax_lims=(-150., -190.),
                 vinf_lims=(-70., -30.))
    gc2.set_sig_v(sig_x_lims=(0.4, 0.3),
                  sig_y_lims=(0.3, 0.4),
                  alpha_lims=(2.0, 1.5),
                  sig_v_in_lims=(70., 90.),
                  sig_v_out_lims=(10., 20.))
    gc2.evaluate_ybar()
    return gc2

@pytest.fixture(scope="module")
def my_stream_component():
    ssps = pkm.model_grids.milesSSPs()
    ssps.logarithmically_resample(dv=100.)
    ssps.calculate_fourier_transform()
    ssps.get_light_weights()
    cube = pkm.ifu_cube.IFUCube(ssps=ssps, nx=9, ny=10)
    stream = pkm.components.stream(cube=cube, rotation=0., center=(0.,0))
    stream.set_p_t(lmd=15., phi=0.3)
    stream.set_p_x(theta_lims=[-np.pi/2., 2.5*np.pi],
                   mu_r_lims=[0.2,0.8],
                   nsmp=1000,
                   sig=0.1)
    stream.set_p_z_t(t_dep=5.)
    stream.set_p_v_x(mu_v_lims=[-100,100], sig_v=110.)
    stream.evaluate_ybar()
    return ssps, cube, stream

@pytest.fixture(scope="module")
def my_three_component_cube(
    my_component,
    my_second_component,
    my_stream_component):
    """Makes a cube with three components
    """
    ssps, cube, gc1 = my_component
    gc2 = my_second_component
    ssps, cube, stream = my_stream_component
    cube.combine_components([gc1, gc2, stream], [0.65, 0.25, 0.1])
    return cube
