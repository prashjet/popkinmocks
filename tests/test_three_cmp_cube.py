import pytest
import matplotlib.pyplot as plt
import numpy as np
import popkinmocks as pkm

@pytest.fixture
def my_ybar():
    ybar = np.array(
        [[[6.67893674e-07, 6.69360939e-07, 6.68723679e-07, 5.83209939e-07,
         5.77660474e-07],
        [8.83664446e-07, 1.08142032e-06, 1.09254773e-06, 9.93580918e-07,
         1.07570487e-06],
        [9.23129628e-07, 1.15894174e-06, 1.71553067e-06, 1.57071179e-06,
         9.68483947e-07]],

       [[4.55575326e-07, 5.72636577e-07, 7.34388803e-07, 6.53639763e-07,
         5.17491328e-07],
        [5.23379075e-07, 7.24404447e-07, 1.11094830e-06, 8.57189741e-07,
         7.84064166e-07],
        [4.75548554e-07, 6.12731175e-07, 1.15137349e-06, 9.49152442e-07,
         5.31339544e-07]],

       [[6.83558330e-07, 8.01792905e-07, 9.40077103e-07, 8.43452419e-07,
         7.15387143e-07],
        [8.29681307e-07, 1.09696934e-06, 1.45891961e-06, 1.19381156e-06,
         1.10976965e-06],
        [8.09503515e-07, 1.03606112e-06, 1.63142567e-06, 1.45142178e-06,
         8.68811094e-07]],

       [[6.24897480e-07, 7.40651866e-07, 8.83779990e-07, 7.83337191e-07,
         6.60060000e-07],
        [7.53447161e-07, 1.00145818e-06, 1.35525034e-06, 1.09919041e-06,
         1.02187853e-06],
        [7.34024910e-07, 9.51035865e-07, 1.54221037e-06, 1.33882259e-06,
         7.88247813e-07]],

       [[5.93968817e-07, 7.11348997e-07, 8.94762578e-07, 7.82693784e-07,
         6.32864908e-07],
        [7.19091297e-07, 9.55536656e-07, 1.34966887e-06, 1.06003987e-06,
         9.94866481e-07],
        [7.15776352e-07, 9.49319357e-07, 1.59209168e-06, 1.34337166e-06,
         7.61759607e-07]],

       [[6.13042484e-07, 7.33139880e-07, 8.91379549e-07, 7.90116482e-07,
         6.53904675e-07],
        [7.34665009e-07, 9.80953392e-07, 1.36428436e-06, 1.09129622e-06,
         1.04115480e-06],
        [7.14785204e-07, 9.28993523e-07, 1.57620235e-06, 1.36950164e-06,
         7.72121085e-07]],

       [[5.90161883e-07, 7.04946526e-07, 8.59582574e-07, 7.58021399e-07,
         6.27472646e-07],
        [7.06923570e-07, 9.43099299e-07, 1.31184921e-06, 1.04784861e-06,
         1.00945780e-06],
        [6.88624051e-07, 8.97063938e-07, 1.54374203e-06, 1.33591919e-06,
         7.44589270e-07]],

       [[5.56292526e-07, 6.61151295e-07, 8.08697513e-07, 7.13126159e-07,
         5.87420168e-07],
        [6.70943574e-07, 8.90973256e-07, 1.23563582e-06, 9.84804192e-07,
         9.57950782e-07],
        [6.60440692e-07, 8.56415215e-07, 1.46089648e-06, 1.28422147e-06,
         7.11763704e-07]],

       [[5.05472788e-07, 6.02644776e-07, 7.60949613e-07, 6.63803978e-07,
         5.34259647e-07],
        [6.12066608e-07, 8.10379986e-07, 1.14871956e-06, 8.97749518e-07,
         8.75354882e-07],
        [6.11906523e-07, 8.02299580e-07, 1.37974925e-06, 1.19452173e-06,
         6.54556964e-07]],

       [[4.94510108e-07, 5.94451647e-07, 7.42040563e-07, 6.50050280e-07,
         5.28070354e-07],
        [5.91601623e-07, 7.89536028e-07, 1.12102144e-06, 8.83428319e-07,
         8.63092741e-07],
        [5.82139251e-07, 7.62962658e-07, 1.34501600e-06, 1.16043791e-06,
         6.28687554e-07]]])


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
    print(ybar.shape)
    assert np.allclose(cube.ybar[::100,::3,::2], ybar)


# end
