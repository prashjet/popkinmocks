import pytest
import matplotlib.pyplot as plt
import numpy as np
import popkinmocks as pkm

@pytest.fixture
def my_ybar():
    ybar = np.array([
        [[5.76763530e-07, 6.01458100e-07, 7.02542999e-07, 5.67799694e-07,
         5.01801101e-07],
        [8.52188445e-07, 1.22565373e-06, 1.23947863e-06, 1.22571712e-06,
         8.10549756e-07],
        [8.94929326e-07, 1.24287066e-06, 1.58971033e-06, 1.38399919e-06,
         9.93859820e-07]],
       [[4.44128893e-07, 5.61224756e-07, 7.54600370e-07, 6.39729406e-07,
         4.99651610e-07],
        [5.02576737e-07, 8.42308960e-07, 1.24471441e-06, 1.04848845e-06,
         5.82878480e-07],
        [4.42256874e-07, 6.70794646e-07, 1.05519622e-06, 7.62919061e-07,
         5.33204940e-07]],
       [[6.42590431e-07, 7.69613804e-07, 9.77107772e-07, 8.30953515e-07,
         6.74716425e-07],
        [7.98194672e-07, 1.24451237e-06, 1.65110468e-06, 1.43817636e-06,
         8.57431701e-07],
        [7.74504667e-07, 1.11876587e-06, 1.51034324e-06, 1.23896257e-06,
         8.79681573e-07]],
       [[5.90123150e-07, 7.12761999e-07, 9.20536193e-07, 7.71744084e-07,
         6.23826138e-07],
        [7.23524911e-07, 1.14803888e-06, 1.53856385e-06, 1.33618683e-06,
         7.84514027e-07],
        [7.02040633e-07, 1.03296432e-06, 1.42257816e-06, 1.13599519e-06,
         7.98392972e-07]],
       [[5.59762990e-07, 6.87930118e-07, 9.33738468e-07, 7.72264224e-07,
         6.02731789e-07],
        [6.86639267e-07, 1.10758284e-06, 1.53479734e-06, 1.30316166e-06,
         7.42511867e-07],
        [6.90575711e-07, 1.04136106e-06, 1.46708159e-06, 1.13326096e-06,
         7.77261319e-07]],
       [[5.77502174e-07, 7.05417213e-07, 9.30406743e-07, 7.77774603e-07,
         6.17843876e-07],
        [7.02066315e-07, 1.13850200e-06, 1.56050376e-06, 1.35211927e-06,
         7.67338494e-07],
        [6.80116001e-07, 1.01798348e-06, 1.44151876e-06, 1.13150253e-06,
         7.83627180e-07]],
       [[5.53351592e-07, 6.75834127e-07, 8.99438144e-07, 7.45588500e-07,
         5.90240608e-07],
        [6.74327999e-07, 1.10252717e-06, 1.50744599e-06, 1.30920897e-06,
         7.35825001e-07],
        [6.53821971e-07, 9.86490254e-07, 1.40697668e-06, 1.09698043e-06,
         7.56693716e-07]],
       [[5.17110354e-07, 6.31431446e-07, 8.46956289e-07, 7.01371442e-07,
         5.50585858e-07],
        [6.38463579e-07, 1.04324053e-06, 1.42511682e-06, 1.23578022e-06,
         6.89748284e-07],
        [6.28313000e-07, 9.44758272e-07, 1.32384470e-06, 1.05452237e-06,
         7.25887992e-07]],
       [[4.68086246e-07, 5.76421627e-07, 7.96449602e-07, 6.53129733e-07,
         5.02306110e-07],
        [5.80058020e-07, 9.53703388e-07, 1.32450324e-06, 1.13058568e-06,
         6.23136082e-07],
        [5.85829801e-07, 8.89405375e-07, 1.24601866e-06, 9.84595238e-07,
         6.70112532e-07]],
       [[4.59521171e-07, 5.68027634e-07, 7.78898970e-07, 6.39172465e-07,
         4.95076786e-07],
        [5.60320142e-07, 9.33777479e-07, 1.30077580e-06, 1.12024216e-06,
         6.10770147e-07],
        [5.51894139e-07, 8.45731370e-07, 1.21530361e-06, 9.42981804e-07,
         6.41288734e-07]]])
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
