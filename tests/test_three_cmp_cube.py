import pytest
import matplotlib.pyplot as plt
import numpy as np
import popkinmocks as pkm

@pytest.fixture
def my_ybar():
    ybar = np.array([[[6.67893679e-07, 6.69361065e-07, 6.68738527e-07, 5.83206367e-07,
             5.77650854e-07],
            [8.83739761e-07, 1.22967264e-06, 1.11781204e-06, 1.19914848e-06,
             8.67235418e-07],
            [9.23146078e-07, 1.25178273e-06, 1.49774684e-06, 1.35238222e-06,
             1.00660348e-06]],

           [[4.55575329e-07, 5.72636680e-07, 7.34400782e-07, 6.53636305e-07,
             5.17482230e-07],
            [5.23439842e-07, 8.44024490e-07, 1.13133326e-06, 1.02700320e-06,
             5.92141161e-07],
            [4.75561811e-07, 6.87571571e-07, 9.70947472e-07, 7.59424800e-07,
             5.61980062e-07]],

           [[6.83558335e-07, 8.01793039e-07, 9.40092605e-07, 8.43448596e-07,
             7.15376885e-07],
            [8.29759948e-07, 1.25177026e-06, 1.48529989e-06, 1.40976275e-06,
             8.88258212e-07],
            [8.09520702e-07, 1.13311416e-06, 1.40244062e-06, 1.22080438e-06,
             9.08727710e-07]],

           [[6.24897485e-07, 7.40651998e-07, 8.83795329e-07, 7.83333562e-07,
             6.60050242e-07],
            [7.53524972e-07, 1.15462636e-06, 1.38135238e-06, 1.30744005e-06,
             8.11390398e-07],
            [7.34041859e-07, 1.04656908e-06, 1.32211961e-06, 1.11977800e-06,
             8.27163602e-07]],

           [[5.93968821e-07, 7.11349138e-07, 8.94779001e-07, 7.82689902e-07,
             6.32854495e-07],
            [7.19174600e-07, 1.11951845e-06, 1.37761370e-06, 1.27927532e-06,
             7.71756376e-07],
            [7.15794456e-07, 1.05123753e-06, 1.36087788e-06, 1.11404449e-06,
             8.02958602e-07]],

           [[6.13042488e-07, 7.33140022e-07, 8.91396251e-07, 7.90112193e-07,
             6.53893223e-07],
            [7.34749726e-07, 1.14771975e-06, 1.39270372e-06, 1.32494399e-06,
             7.95690563e-07],
            [7.14803721e-07, 1.03358343e-06, 1.32830935e-06, 1.11761670e-06,
             8.15110562e-07]],

           [[5.90161887e-07, 7.04946670e-07, 8.59599535e-07, 7.58017158e-07,
             6.27461286e-07],
            [7.07009604e-07, 1.11245766e-06, 1.34071029e-06, 1.28362664e-06,
             7.65220218e-07],
            [6.88642847e-07, 1.00320214e-06, 1.29379263e-06, 1.08365428e-06,
             7.88163412e-07]],

           [[5.56292530e-07, 6.61151435e-07, 8.08714002e-07, 7.13122066e-07,
             5.87409192e-07],
            [6.71027211e-07, 1.05561329e-06, 1.26369282e-06, 1.21430184e-06,
             7.21626152e-07],
            [6.60458972e-07, 9.59664434e-07, 1.21755088e-06, 1.03919652e-06,
             7.54210355e-07]],

           [[5.05472791e-07, 6.02644909e-07, 7.60965440e-07, 6.63800143e-07,
             5.34249367e-07],
            [6.12146886e-07, 9.68410622e-07, 1.17565024e-06, 1.11276810e-06,
             6.55151528e-07],
            [6.11924012e-07, 9.00902820e-07, 1.15242694e-06, 9.67755880e-07,
             6.94708774e-07]],

           [[4.94510111e-07, 5.94451779e-07, 7.42056310e-07, 6.50046423e-07,
             5.28060003e-07],
            [5.91681489e-07, 9.46755735e-07, 1.14781392e-06, 1.10061781e-06,
             6.40363019e-07],
            [5.82156685e-07, 8.61367903e-07, 1.11498622e-06, 9.29660577e-07,
             6.69001405e-07]]])

    return ybar


def test_three_component_cube_normalisations(my_three_component_cube):
    """Checks value of ybar and kurtosis map for a two component mixture model

    """
    cube = my_three_component_cube
    ssps = cube.ssps
    v_edg = np.linspace(-1000, 1000, 50)
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
    a = cube.get_p('vx', v_edg=v_edg, density=False, light_weighted=False, collapse_cmps=True)
    assert np.allclose(np.sum(a), 1.)
    a = cube.get_p('vx', v_edg=v_edg, density=True, light_weighted=False, collapse_cmps=True)
    assert np.allclose(np.sum(a*dv*cube.dx*cube.dy), 1.)
    a = cube.get_p('vx', v_edg=v_edg, density=False, light_weighted=True, collapse_cmps=True)
    assert np.allclose(np.sum(a), 1.)
    a = cube.get_p('vx', v_edg=v_edg, density=True, light_weighted=True, collapse_cmps=True)
    assert np.allclose(np.sum(a*dv*cube.dx*cube.dy), 1.)
    # check get_p_tvxz
    a = cube.get_p('tvxz', v_edg=v_edg, density=False, light_weighted=False, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1)
    a = cube.get_p('tvxz', v_edg=v_edg, density=True, light_weighted=False, collapse_cmps=True)
    vol_elmt = ssps.delta_t[:,na,na,na,na]*ssps.delta_z[na,na,na,na,:]
    vol_elmt *= cube.dx * cube.dy * dv
    assert np.isclose(np.sum(a*vol_elmt), 1)
    a = cube.get_p('tvxz', v_edg=v_edg, density=False, light_weighted=True, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1)
    a = cube.get_p('tvxz', v_edg=v_edg, density=True, light_weighted=True, collapse_cmps=True)
    vol_elmt = ssps.delta_t[:,na,na,na,na]*ssps.delta_z[na,na,na,na,:]
    vol_elmt *= cube.dx * cube.dy * dv
    assert np.isclose(np.sum(a*vol_elmt), 1)
    # check get_p_v_x
    a = cube.get_p('v_x', v_edg=v_edg, density=False, light_weighted=False, collapse_cmps=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = cube.get_p('v_x', v_edg=v_edg, density=True, light_weighted=False, collapse_cmps=True)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    a = cube.get_p('v_x', v_edg=v_edg, density=False, light_weighted=True, collapse_cmps=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = cube.get_p('v_x', v_edg=v_edg, density=True, light_weighted=True, collapse_cmps=True)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    # check get_p_v
    a = cube.get_p('v', v_edg=v_edg, density=False, light_weighted=False, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1.)
    a = cube.get_p('v', v_edg=v_edg, density=True, light_weighted=False, collapse_cmps=True)
    assert np.isclose(np.sum(a*dv), 1.)
    a = cube.get_p('v', v_edg=v_edg, density=False, light_weighted=True, collapse_cmps=True)
    assert np.isclose(np.sum(a), 1.)
    a = cube.get_p('v', v_edg=v_edg, density=True, light_weighted=True, collapse_cmps=True)
    assert np.isclose(np.sum(a*dv), 1.)


def test_three_component_cube_ybar(my_three_component_cube, my_ybar):
    cube = my_three_component_cube
    ybar = my_ybar
    print(ybar.shape)
    assert np.allclose(cube.ybar[::100,::3,::2], ybar)


# end
