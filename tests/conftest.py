import pytest
import matplotlib.pyplot as plt
import numpy as np
import popkinmocks as pkm
from itertools import chain, combinations

N_VELOCITY_BINS = 50
VELOCITY_LIMIT = 1000
NX, NY = 4,5
EVAL_YBAR = True

def my_cube(nv=N_VELOCITY_BINS, nx=NX, ny=NY, vlim=VELOCITY_LIMIT):
    v_edg = np.linspace(-vlim, vlim, nv)
    ssps = pkm.model_grids.milesSSPs()
    ssps.logarithmically_resample(dv=v_edg[1]-v_edg[0])
    ssps.calculate_fourier_transform()
    ssps.get_light_weights()
    cube = pkm.ifu_cube.IFUCube(ssps=ssps, nx=nx, ny=ny, v_edg=v_edg)
    return cube

@pytest.fixture(scope="module", name='my_cube')
def my_cube_fixture():
    return my_cube()

def my_component(eval_ybar=EVAL_YBAR, cube=my_cube()):
    gc1 = pkm.components.GrowingDisk(cube=cube, rotation=0., center=(0.2,0))
    gc1.set_p_t(lmd=2., phi=0.8)
    gc1.set_p_x_t(q_lims=(0.05, 0.5),
                  rc_lims=(1., 1.),
                  alpha_lims=(1.2, 0.8))
    gc1.set_t_dep(q=0.2,
                  alpha=3.,
                  t_dep_in=0.5,
                  t_dep_out=5.)
    gc1.set_p_z_tx()
    gc1.set_mu_v(q_lims=(0.5, 0.1),
                 rmax_lims=(0.5, 1.1),
                 vmax_lims=(250., 100.))
    gc1.set_sig_v(q_lims=(0.25, 1.0),
                  alpha_lims=(3.0, 2.5),
                  sig_v_in_lims=(70., 50.),
                  sig_v_out_lims=(80., 60.))
    if eval_ybar:
        gc1.evaluate_ybar()
    return gc1

@pytest.fixture(scope="module", name='my_component')
def my_component_fixture(my_cube):
    return my_component(my_cube)

def my_second_component(eval_ybar=EVAL_YBAR, cube=my_cube()):
    gc2 = pkm.components.GrowingDisk(cube=cube,
                                     rotation=np.deg2rad(10.),
                                     center=(0.05,-0.07))
    gc2.set_p_t(lmd=1.6, phi=0.3)
    gc2.set_p_x_t(q_lims=(1., 1.),
                  rc_lims=(1., 1.),
                  alpha_lims=(1.1, 1.1))
    gc2.set_t_dep(q=0.7,
                  alpha=1.1,
                  t_dep_in=6.,
                  t_dep_out=1.)
    gc2.set_p_z_tx()
    gc2.set_mu_v(q_lims=(0.5, 0.4),
                 rmax_lims=(0.7, 0.2),
                 vmax_lims=(-150., -190.))
    gc2.set_sig_v(q_lims=(0.3/0.4, 0.4/0.3),
                  alpha_lims=(2.0, 1.5),
                  sig_v_in_lims=(70., 90.),
                  sig_v_out_lims=(10., 20.))
    if eval_ybar:
        gc2.evaluate_ybar()
    return gc2

@pytest.fixture(scope="module", name='my_second_component')
def my_second_component_fixture(my_cube):
    return my_second_component(my_cube)

def my_stream_component(eval_ybar=EVAL_YBAR, cube=my_cube()):
    stream = pkm.components.Stream(cube=cube, rotation=0., center=(0.,0))
    stream.set_p_t(lmd=15., phi=0.3)
    stream.set_p_x_t(theta_lims=[-np.pi/2., 0.75*np.pi],
                     mu_r_lims=[0.2,0.8],
                     nsmp=1000,
                     sig=0.1)
    stream.set_t_dep(t_dep=5.)
    stream.set_p_z_tx()
    stream.set_mu_v(mu_v_lims=[-80,100])
    stream.set_sig_v(sig_v=110.)
    if eval_ybar:
        stream.evaluate_ybar()
    return stream

@pytest.fixture(scope="module", name='my_stream_component')
def my_stream_component_fixture(my_cube):
    return my_stream_component(my_cube)

def my_galaxy(eval_ybar=EVAL_YBAR, cube=my_cube()):
    gc1 = my_component(eval_ybar=eval_ybar, cube=cube)
    gc2 = my_second_component(eval_ybar=eval_ybar, cube=cube)
    stream = my_stream_component(eval_ybar=eval_ybar, cube=cube)
    galaxy = pkm.components.Mixture(
        cube=cube,
        component_list=[gc1, gc2, stream],
        weights=[0.65, 0.25, 0.1])
    if eval_ybar:
        galaxy.evaluate_ybar()
    return galaxy

@pytest.fixture(scope="module", name="my_galaxy")
def my_galaxy_fixture(my_cube):
    galaxy = my_galaxy(my_cube)
    return galaxy

def my_base_component(cube, galaxy, eval_ybar=EVAL_YBAR):
    log_p_tvxz = galaxy.get_log_p(
        'tvxz',density=True,
        light_weighted=False,
        collapse_cmps=True)
    base_cmp = pkm.components.base.Component(cube=cube, log_p_tvxz=log_p_tvxz)
    if eval_ybar:
        base_cmp.evaluate_ybar()
    return base_cmp

@pytest.fixture(scope="module", name="my_base_component")
def my_base_component_fixture(my_cube, my_galaxy):
    base_cmp = my_base_component(my_cube, my_galaxy)
    return base_cmp

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    poset = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return list(poset)

def distribution_list():
    "get a list of all possible distributions involving the 4 pop-kin variables"
    pkvars = ['t', 'v', 'x', 'z']
    poset = powerset(pkvars)
    subset_sizes = np.array([len(subset) for subset in poset])
    # list possible combinations of numbers of independent/dependent variables
    n_vars_in_dist = []
    for n_depend in [1,2,3,4]:
        for n_indep in np.arange(5-n_depend):
            n_vars_in_dist += [(n_depend, n_indep)]
    pk_distributions = []
    for (nd, ni) in n_vars_in_dist:
        idx_d = np.where(subset_sizes == nd)[0]
        for idx_d0 in idx_d:
            idx_i = np.where(subset_sizes == ni)[0]
            for idx_i0 in idx_i:
                # independent/dependent variables must be disjoint sets
                if set(poset[idx_d0]).intersection(set(poset[idx_i0]))==set():
                    str_d = ''.join(sorted(poset[idx_d0]))
                    str_i = ''.join(sorted(poset[idx_i0]))
                    if str_i=='':
                        pk_distributions += [str_d]
                    else:
                        pk_distributions += [f'{str_d}_{str_i}']
    return pk_distributions

@pytest.fixture(scope="module", name='distribution_list')
def distribution_list_fixture():
    return distribution_list()
