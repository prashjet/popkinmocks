import pytest
import matplotlib.pyplot as plt
import numpy as np
import popkinmocks as pkm

@pytest.fixture
def my_stream_ybar_trim():
    ybar_trim = np.array([[
        [3.58595281e-36, 6.33020057e-27, 2.09853431e-18, 4.41097781e-11,
         9.89185428e-11],
        [9.95940956e-22, 1.22461802e-12, 6.16495353e-11, 1.20986102e-07,
         1.03773979e-06],
        [1.44113779e-16, 8.83602320e-08, 2.64751679e-06, 1.09872837e-07,
         2.19658812e-06],
        [6.03784963e-18, 8.02469484e-09, 3.32021154e-06, 3.23370629e-06,
         7.56592089e-08],
        [2.51340408e-26, 7.57487921e-16, 1.89172193e-11, 3.37473421e-11,
         1.06636959e-14]],

       [[4.14736554e-36, 7.32124963e-27, 2.42707848e-18, 4.89963590e-11,
         1.09876963e-10],
        [1.15186435e-21, 1.41634284e-12, 7.13013170e-11, 1.34389216e-07,
         1.15457569e-06],
        [1.66676069e-16, 1.02193810e-07, 3.06200904e-06, 1.22941379e-07,
         2.45785563e-06],
        [6.96835623e-18, 9.24336629e-09, 3.80095247e-06, 3.66416077e-06,
         8.52610880e-08],
        [2.89383651e-26, 8.69881872e-16, 2.16288809e-11, 3.83751665e-11,
         1.20713605e-14]],

       [[3.69023890e-36, 6.51429441e-27, 2.15956354e-18, 4.48110908e-11,
         1.00491274e-10],
        [1.02490475e-21, 1.26023216e-12, 6.34424168e-11, 1.22909692e-07,
         1.05145746e-06],
        [1.48304872e-16, 9.09299096e-08, 2.72451144e-06, 1.10783304e-07,
         2.21479025e-06],
        [6.17744701e-18, 8.18310983e-09, 3.36295807e-06, 3.25755263e-06,
         7.61608297e-08],
        [2.56141056e-26, 7.69505286e-16, 1.91418268e-11, 3.40391471e-11,
         1.07397321e-14]],

       [[3.88457784e-36, 6.85735651e-27, 2.27329257e-18, 5.10110490e-11,
         1.14395013e-10],
        [1.07887927e-21, 1.32659973e-12, 6.67834830e-11, 1.39915190e-07,
         1.19384384e-06],
        [1.56115047e-16, 9.57185487e-08, 2.86799230e-06, 1.24572600e-07,
         2.49046717e-06],
        [6.54982019e-18, 8.71856963e-09, 3.62915656e-06, 3.58251253e-06,
         8.45807390e-08],
        [2.73175086e-26, 8.25268613e-16, 2.07071999e-11, 3.71965211e-11,
         1.18318078e-14]],

       [[4.06475549e-36, 7.17542000e-27, 2.37873427e-18, 5.18963341e-11,
         1.16380312e-10],
        [1.12892073e-21, 1.38813116e-12, 6.98810888e-11, 1.42343387e-07,
         1.21604963e-06],
        [1.63356102e-16, 1.00158244e-07, 3.00101786e-06, 1.27479535e-07,
         2.54858289e-06],
        [6.85315285e-18, 9.11532331e-09, 3.78061335e-06, 3.70397828e-06,
         8.70637458e-08],
        [2.85546266e-26, 8.61414665e-16, 2.15528588e-11, 3.85628547e-11,
         1.22235109e-14]],

       [[4.01871211e-36, 7.09414066e-27, 2.35178924e-18, 5.16670477e-11,
         1.15866125e-10],
        [1.11613292e-21, 1.37240715e-12, 6.90895131e-11, 1.41714490e-07,
         1.21094419e-06],
        [1.61505691e-16, 9.90237051e-08, 2.96702393e-06, 1.26973206e-07,
         2.53846031e-06],
        [6.78530508e-18, 9.03334999e-09, 3.75432358e-06, 3.68573037e-06,
         8.66883040e-08],
        [2.83029481e-26, 8.54629773e-16, 2.14099820e-11, 3.83513544e-11,
         1.21648980e-14]],

       [[3.77494524e-36, 6.66382459e-27, 2.20913451e-18, 4.73349287e-11,
         1.06151116e-10],
        [1.04843058e-21, 1.28915973e-12, 6.48986845e-11, 1.29832177e-07,
         1.10957300e-06],
        [1.51709086e-16, 9.30171296e-08, 2.78705032e-06, 1.16517652e-07,
         2.32943188e-06],
        [6.35224117e-18, 8.43846764e-09, 3.48855791e-06, 3.40205484e-06,
         7.97867004e-08],
        [2.64275255e-26, 7.96115422e-16, 1.98758234e-11, 3.54738371e-11,
         1.12224863e-14]]])

    return ybar_trim

def test_component_normalisation(my_stream_component):
    """Tests the normalisations of all densities evaluated for a component

    """
    ssps, cube, stream = my_stream_component
    v_edg = cube.v_edg
    dv = v_edg[1] - v_edg[0]
    na = np.newaxis
    # check p_t
    a = stream.get_p_t(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1)
    a = stream.get_p_t(density=True, light_weighted=False)
    assert np.isclose(np.sum(a*ssps.delta_t), 1)
    a = stream.get_p_t(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1)
    a = stream.get_p_t(density=True, light_weighted=True)
    assert np.isclose(np.sum(a*ssps.delta_t), 1)
    # check p_x_t
    a = stream.get_p_x_t(density=False, light_weighted=False)
    assert np.allclose(np.sum(a, (0,1)), 1.)
    a = stream.get_p_x_t(density=True, light_weighted=False)
    assert np.allclose(np.sum(a*cube.dx*cube.dy, (0,1)), 1.)
    a = stream.get_p_x_t(density=False, light_weighted=True)
    assert np.allclose(np.sum(a, (0,1)), 1.)
    a = stream.get_p_x_t(density=True, light_weighted=True)
    assert np.allclose(np.sum(a*cube.dx*cube.dy, (0,1)), 1.)
    # check p_tx
    a = stream.get_p_tx(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1.)
    a = stream.get_p_tx(density=True, light_weighted=False)
    assert np.isclose(np.sum(a*ssps.delta_t[:,na,na]*cube.dx*cube.dy), 1)
    a = stream.get_p_tx(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1.)
    a = stream.get_p_tx(density=True, light_weighted=True)
    assert np.isclose(np.sum(a*ssps.delta_t[:,na,na]*cube.dx*cube.dy), 1)
    # check p_z_tx
    a = stream.get_p_z_tx(density=False, light_weighted=False)
    assert np.allclose(np.sum(a, 0), 1.)
    a = stream.get_p_z_tx(density=True, light_weighted=False)
    assert np.allclose(np.sum(a*ssps.delta_z[:,na,na,na], 0), 1.)
    a = stream.get_p_z_tx(density=False, light_weighted=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = stream.get_p_z_tx(density=True, light_weighted=True)
    assert np.allclose(np.sum(a*ssps.delta_z[:,na,na,na], 0), 1.)
    # # check p_z_t
    # a = stream.get_p_z_t(density=False, light_weighted=False)
    # assert np.allclose(np.sum(a, 0), 1.)
    # a = stream.get_p_z_t(density=True, light_weighted=False)
    # assert np.allclose(np.sum(a*ssps.delta_z[:,na], 0), 1.)
    # a = stream.get_p_z_t(density=False, light_weighted=True)
    # assert np.allclose(np.sum(a, 0), 1.)
    # a = stream.get_p_z_t(density=True, light_weighted=True)
    # assert np.allclose(np.sum(a*ssps.delta_z[:,na], 0), 1.)
    # check p_txz
    a = stream.get_p_txz(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1)
    a = stream.get_p_txz(density=True, light_weighted=False)
    vol_elmt = ssps.delta_t[:,na,na,na]*cube.dx*cube.dy*ssps.delta_z[na,na,na,:]
    assert np.isclose(np.sum(a * vol_elmt), 1)
    a = stream.get_p_txz(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1)
    a = stream.get_p_txz(density=True, light_weighted=True)
    vol_elmt = ssps.delta_t[:,na,na,na]*cube.dx*cube.dy*ssps.delta_z[na,na,na,:]
    assert np.isclose(np.sum(a * vol_elmt), 1)
    # check get_p_x
    a = stream.get_p_x(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1)
    a = stream.get_p_x(density=True, light_weighted=False)
    assert np.isclose(np.sum(a*cube.dx*cube.dy), 1)
    a = stream.get_p_x(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1)
    a = stream.get_p_x(density=True, light_weighted=True)
    assert np.isclose(np.sum(a*cube.dx*cube.dy), 1)
    # check get_p_z
    a = stream.get_p_z(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1)
    a = stream.get_p_z(density=True, light_weighted=False)
    assert np.isclose(np.sum(a*ssps.delta_z), 1)
    a = stream.get_p_z(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1)
    a = stream.get_p_z(density=True, light_weighted=True)
    assert np.isclose(np.sum(a*ssps.delta_z), 1)
    # check p_tz
    a = stream.get_p_tz(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1.)
    a = stream.get_p_tz(density=True, light_weighted=False)
    assert np.isclose(np.sum(a*ssps.delta_t[:,na]*ssps.delta_z[na,:]), 1.)
    a = stream.get_p_tz(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1.)
    a = stream.get_p_tz(density=True, light_weighted=True)
    assert np.isclose(np.sum(a*ssps.delta_t[:,na]*ssps.delta_z[na,:]), 1.)
    # check get_p_v
    a = stream.get_p_v(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1.)
    a = stream.get_p_v(density=True, light_weighted=False)
    assert np.isclose(np.sum(a*dv), 1.)
    a = stream.get_p_v(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1.)
    a = stream.get_p_v(density=True, light_weighted=True)
    assert np.isclose(np.sum(a*dv), 1.)
    # check get_p_tvxz
    a = stream.get_p_tvxz(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1)
    a = stream.get_p_tvxz(density=True, light_weighted=False)
    vol_elmt = ssps.delta_t[:,na,na,na,na]*ssps.delta_z[na,na,na,na,:]
    vol_elmt *= cube.dx * cube.dy * dv
    assert np.isclose(np.sum(a*vol_elmt), 1)
    a = stream.get_p_tvxz(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1)
    a = stream.get_p_tvxz(density=True, light_weighted=True)
    vol_elmt = ssps.delta_t[:,na,na,na,na]*ssps.delta_z[na,na,na,na,:]
    vol_elmt *= cube.dx * cube.dy * dv
    assert np.isclose(np.sum(a*vol_elmt), 1)
    # check get_p_v_tx
    a = stream.get_p('v_tx', density=False, light_weighted=False)
    assert np.allclose(np.sum(a, 0), 1.)
    a = stream.get_p('v_tx', density=True, light_weighted=False)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    a = stream.get_p('v_tx', density=False, light_weighted=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = stream.get_p('v_tx', density=True, light_weighted=True)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    # check p_tz_x
    a = stream.get_p('tz_x', density=False, light_weighted=False)
    assert np.allclose(np.sum(a, (0,1)), 1.)
    a = stream.get_p('tz_x', density=True, light_weighted=False)
    vol_elmt = ssps.delta_t[:,na,na,na]*ssps.delta_z[na,:,na,na]
    assert np.allclose(np.sum(a*vol_elmt, (0,1)), 1.)
    a = stream.get_p('tz_x', density=False, light_weighted=True)
    assert np.allclose(np.sum(a, (0,1)), 1.)
    a = stream.get_p('tz_x', density=True, light_weighted=True)
    vol_elmt = ssps.delta_t[:,na,na,na]*ssps.delta_z[na,:,na,na]
    assert np.allclose(np.sum(a*vol_elmt, (0,1)), 1.)
    # check get_p_v_x
    a = stream.get_p('v_x', density=False, light_weighted=False)
    assert np.allclose(np.sum(a, 0), 1.)
    a = stream.get_p('v_x', density=True, light_weighted=False)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    a = stream.get_p('v_x', density=False, light_weighted=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = stream.get_p('v_x', density=True, light_weighted=True)
    assert np.allclose(np.sum(a*dv, 0), 1.)

def test_component_ybar(
    my_stream_component,
    my_stream_ybar_trim,
    ):
    """Checks value of ybar for one component

    """
    ssps, cube, stream = my_stream_component
    ybar_trim = stream.ybar[::150, ::2, ::2]
    assert np.allclose(ybar_trim, my_stream_ybar_trim)


# end
