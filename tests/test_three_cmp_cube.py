import pytest
import matplotlib.pyplot as plt
import numpy as np
import popkinmocks as pkm

@pytest.fixture
def my_ybar():
    ybar = np.array([[[6.61713225e-07, 6.57574889e-07, 6.65988628e-07, 5.80388448e-07,
         5.66542885e-07],
        [8.81048640e-07, 1.07286117e-06, 1.08458991e-06, 9.79574342e-07,
         1.07375293e-06],
        [9.23120707e-07, 1.15513879e-06, 1.72492828e-06, 1.57798473e-06,
         9.68661650e-07]],

       [[6.72685245e-07, 7.97262398e-07, 9.34860013e-07, 8.44367565e-07,
         7.12474108e-07],
        [8.10178467e-07, 1.07905773e-06, 1.45876887e-06, 1.18866128e-06,
         1.11394096e-06],
        [7.83185408e-07, 9.98325611e-07, 1.59430946e-06, 1.44247527e-06,
         8.45613580e-07]],

       [[6.54917834e-07, 7.71838363e-07, 9.30504260e-07, 8.29908338e-07,
         6.87430769e-07],
        [7.98975527e-07, 1.05487889e-06, 1.43256327e-06, 1.15121472e-06,
         1.06968925e-06],
        [7.92998778e-07, 1.03560683e-06, 1.64769224e-06, 1.43153722e-06,
         8.42853110e-07]],

       [[5.42068761e-07, 6.52938174e-07, 8.06288595e-07, 7.26663426e-07,
         5.84775769e-07],
        [6.54617933e-07, 8.73399565e-07, 1.23631450e-06, 9.76035331e-07,
         8.88786050e-07],
        [6.47364760e-07, 8.59070326e-07, 1.38525422e-06, 1.17226996e-06,
         6.87316724e-07]],

       [[6.43073563e-07, 7.62811990e-07, 9.20152213e-07, 8.17782904e-07,
         6.79471304e-07],
        [7.76794039e-07, 1.03094537e-06, 1.41220556e-06, 1.13622595e-06,
         1.07719171e-06],
        [7.61254032e-07, 9.83670960e-07, 1.61590061e-06, 1.43060941e-06,
         8.18186769e-07]],

       [[6.04675017e-07, 7.19325954e-07, 8.61208500e-07, 7.62702523e-07,
         6.41031564e-07],
        [7.23708549e-07, 9.65884288e-07, 1.32236775e-06, 1.06790917e-06,
         1.02270638e-06],
        [6.98821914e-07, 9.05081454e-07, 1.53962782e-06, 1.33498974e-06,
         7.57222728e-07]],

       [[5.86219763e-07, 7.01686121e-07, 8.51797124e-07, 7.55149111e-07,
         6.25688041e-07],
        [7.00838870e-07, 9.36882542e-07, 1.30358950e-06, 1.04354573e-06,
         1.00505071e-06],
        [6.79601034e-07, 8.83204006e-07, 1.52000681e-06, 1.31914071e-06,
         7.36179282e-07]],

       [[5.69584411e-07, 6.77191608e-07, 8.24564230e-07, 7.26459816e-07,
         6.02284883e-07],
        [6.84098294e-07, 9.10551274e-07, 1.25921867e-06, 1.00749437e-06,
         9.81479728e-07],
        [6.66440450e-07, 8.61552365e-07, 1.48532965e-06, 1.30082578e-06,
         7.21705197e-07]],

       [[5.57874569e-07, 6.68602709e-07, 8.16771476e-07, 7.21082053e-07,
         5.95468641e-07],
        [6.66473462e-07, 8.90707181e-07, 1.24548298e-06, 9.94209672e-07,
         9.73011598e-07],
        [6.48459882e-07, 8.40786647e-07, 1.47100058e-06, 1.28749358e-06,
         7.03598859e-07]],

       [[4.96223656e-07, 5.93474999e-07, 7.25214697e-07, 6.37540536e-07,
         5.26883336e-07],
        [5.94045967e-07, 7.92357013e-07, 1.10369132e-06, 8.80442521e-07,
         8.62877733e-07],
        [5.82960741e-07, 7.65077315e-07, 1.33393264e-06, 1.15744612e-06,
         6.29391004e-07]],

       [[5.16080343e-07, 6.15466495e-07, 7.39659427e-07, 6.52891835e-07,
         5.48356640e-07],
        [6.10495496e-07, 8.18189508e-07, 1.13128489e-06, 9.11922700e-07,
         9.05655983e-07],
        [5.79174974e-07, 7.40158390e-07, 1.33411939e-06, 1.16835043e-06,
         6.38196302e-07]]])


    return ybar


def test_three_component_cube_normalisations(my_three_component_cube):
    """Checks value of ybar and kurtosis map for a two component mixture model

    """
    cube = my_three_component_cube
    ssps = cube.ssps
    v_edg = cube.v_edg
    dv = v_edg[1] - v_edg[0]
    na = np.newaxis
    # check p_t
    a = cube.get_p('t', density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1)
    a = cube.get_p('t', density=True, light_weighted=False)
    assert np.isclose(np.sum(a*ssps.delta_t), 1)
    a = cube.get_p('t', density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1)
    a = cube.get_p('t', density=True, light_weighted=True)
    assert np.isclose(np.sum(a*ssps.delta_t), 1)
    # check p_x_t
    a = cube.get_p('x_t', density=False, light_weighted=False, collapse_cmps=True)
    assert np.allclose(np.sum(a, (0,1)), 1.)
    a = cube.get_p('x_t', density=True, light_weighted=False, collapse_cmps=True)
    assert np.allclose(np.sum(a*cube.dx*cube.dy, (0,1)), 1.)
    a = cube.get_p('x_t', density=False, light_weighted=True, collapse_cmps=True)
    assert np.allclose(np.sum(a, (0,1)), 1.)
    a = cube.get_p('x_t', density=True, light_weighted=True, collapse_cmps=True)
    assert np.allclose(np.sum(a*cube.dx*cube.dy, (0,1)), 1.)
    # check p_tx
    a = cube.get_p('tx', density=False, light_weighted=False, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1.)
    a = cube.get_p('tx', density=True, light_weighted=False, collapse_cmps=True)
    assert np.isclose(np.sum(a*ssps.delta_t[:,na,na]*cube.dx*cube.dy), 1)
    a = cube.get_p('tx', density=False, light_weighted=True, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1.)
    a = cube.get_p('tx', density=True, light_weighted=True, collapse_cmps=True)
    assert np.isclose(np.sum(a*ssps.delta_t[:,na,na]*cube.dx*cube.dy), 1)
    # check p_z_tx
    a = cube.get_p('z_tx', density=False, light_weighted=False, collapse_cmps=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = cube.get_p('z_tx', density=True, light_weighted=False, collapse_cmps=True)
    assert np.allclose(np.sum(a*ssps.delta_z[:,na,na,na], 0), 1.)
    a = cube.get_p('z_tx', density=False, light_weighted=True, collapse_cmps=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = cube.get_p('z_tx', density=True, light_weighted=True, collapse_cmps=True)
    assert np.allclose(np.sum(a*ssps.delta_z[:,na,na,na], 0), 1.)
    # check p_txz
    a = cube.get_p('txz', density=False, light_weighted=False, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1)
    a = cube.get_p('txz', density=True, light_weighted=False, collapse_cmps=True)
    vol_elmt = ssps.delta_t[:,na,na,na]*cube.dx*cube.dy*ssps.delta_z[na,na,na,:]
    assert np.isclose(np.sum(a * vol_elmt), 1)
    a = cube.get_p('txz', density=False, light_weighted=True, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1)
    a = cube.get_p('txz', density=True, light_weighted=True, collapse_cmps=True)
    vol_elmt = ssps.delta_t[:,na,na,na]*cube.dx*cube.dy*ssps.delta_z[na,na,na,:]
    assert np.isclose(np.sum(a * vol_elmt), 1)
    # check get_p_x
    a = cube.get_p('x', density=False, light_weighted=False, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1)
    a = cube.get_p('x', density=True, light_weighted=False, collapse_cmps=True)
    assert np.isclose(np.sum(a*cube.dx*cube.dy), 1)
    a = cube.get_p('x', density=False, light_weighted=True, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1)
    a = cube.get_p('x', density=True, light_weighted=True, collapse_cmps=True)
    assert np.isclose(np.sum(a*cube.dx*cube.dy), 1)
    # check get_p_z
    a = cube.get_p('z', density=False, light_weighted=False, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1)
    a = cube.get_p('z', density=True, light_weighted=False, collapse_cmps=True)
    assert np.isclose(np.sum(a*ssps.delta_z), 1)
    a = cube.get_p('z', density=False, light_weighted=True, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1)
    a = cube.get_p('z', density=True, light_weighted=True, collapse_cmps=True)
    assert np.isclose(np.sum(a*ssps.delta_z), 1)
    # check p_tz_x
    a = cube.get_p('tz_x', density=False, light_weighted=False, collapse_cmps=True)
    assert np.allclose(np.sum(a, (0,1)), 1.)
    a = cube.get_p('tz_x', density=True, light_weighted=False, collapse_cmps=True)
    vol_elmt = ssps.delta_t[:,na,na,na]*ssps.delta_z[na,:,na,na]
    assert np.allclose(np.sum(a*vol_elmt, (0,1)), 1.)
    a = cube.get_p('tz_x', density=False, light_weighted=True, collapse_cmps=True)
    assert np.allclose(np.sum(a, (0,1)), 1.)
    a = cube.get_p('tz_x', density=True, light_weighted=True, collapse_cmps=True)
    vol_elmt = ssps.delta_t[:,na,na,na]*ssps.delta_z[na,:,na,na]
    assert np.allclose(np.sum(a*vol_elmt, (0,1)), 1.)
    # check p_tz
    a = cube.get_p('tz', density=False, light_weighted=False, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1.)
    a = cube.get_p('tz', density=True, light_weighted=False, collapse_cmps=True)
    assert np.isclose(np.sum(a*ssps.delta_t[:,na]*ssps.delta_z[na,:]), 1.)
    a = cube.get_p('tz', density=False, light_weighted=True, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1.)
    a = cube.get_p('tz', density=True, light_weighted=True, collapse_cmps=True)
    assert np.isclose(np.sum(a*ssps.delta_t[:,na]*ssps.delta_z[na,:]), 1.)
    # # check get_p_vx
    a = cube.get_p('vx', density=False, light_weighted=False, collapse_cmps=True)
    assert np.allclose(np.sum(a), 1.)
    a = cube.get_p('vx', density=True, light_weighted=False, collapse_cmps=True)
    assert np.allclose(np.sum(a*dv*cube.dx*cube.dy), 1.)
    a = cube.get_p('vx', density=False, light_weighted=True, collapse_cmps=True)
    assert np.allclose(np.sum(a), 1.)
    a = cube.get_p('vx', density=True, light_weighted=True, collapse_cmps=True)
    assert np.allclose(np.sum(a*dv*cube.dx*cube.dy), 1.)
    # check get_p_tvxz
    a = cube.get_p('tvxz', density=False, light_weighted=False, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1)
    a = cube.get_p('tvxz', density=True, light_weighted=False, collapse_cmps=True)
    vol_elmt = ssps.delta_t[:,na,na,na,na]*ssps.delta_z[na,na,na,na,:]
    vol_elmt *= cube.dx * cube.dy * dv
    assert np.isclose(np.sum(a*vol_elmt), 1)
    a = cube.get_p('tvxz', density=False, light_weighted=True, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1)
    a = cube.get_p('tvxz', density=True, light_weighted=True, collapse_cmps=True)
    vol_elmt = ssps.delta_t[:,na,na,na,na]*ssps.delta_z[na,na,na,na,:]
    vol_elmt *= cube.dx * cube.dy * dv
    assert np.isclose(np.sum(a*vol_elmt), 1)
    # check get_p_v_x
    a = cube.get_p('v_x', density=False, light_weighted=False, collapse_cmps=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = cube.get_p('v_x', density=True, light_weighted=False, collapse_cmps=True)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    a = cube.get_p('v_x', density=False, light_weighted=True, collapse_cmps=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = cube.get_p('v_x', density=True, light_weighted=True, collapse_cmps=True)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    # check get_p_v
    a = cube.get_p('v', density=False, light_weighted=False, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1.)
    a = cube.get_p('v', density=True, light_weighted=False, collapse_cmps=True)
    assert np.isclose(np.sum(a*dv), 1.)
    a = cube.get_p('v', density=False, light_weighted=True, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1.)
    a = cube.get_p('v', density=True, light_weighted=True, collapse_cmps=True)
    assert np.isclose(np.sum(a*dv), 1.)


def test_three_component_cube_ybar(my_three_component_cube, my_ybar):
    cube = my_three_component_cube
    ybar = my_ybar
    assert np.allclose(cube.ybar[::100,::3,::2], ybar)


def test_datacube_exact_fts_vs_ffts(my_three_component_cube, my_ybar):
    cube = my_three_component_cube
    ybar = my_ybar
    p_tvxz = cube.get_p('tvxz',
                        density=True,
                        light_weighted=False,
                        collapse_cmps=True)
    mix_cmp = pkm.components.Component(cube=cube, p_tvxz=p_tvxz)
    mix_cmp.evaluate_ybar()
    frac_error = (mix_cmp.ybar[::100,::3,::2] - ybar)/ybar
    # check median absolute error < 0.15 %
    # this choice is justified in `error_limit_exactFTs_vs_FFTs.ipynb`
    mad = np.median(np.abs(frac_error))
    assert mad<0.0015








# end
