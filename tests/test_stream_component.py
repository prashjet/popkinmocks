import pytest
import matplotlib.pyplot as plt
import numpy as np
import popkinmocks as pkm

@pytest.fixture
def my_stream_ybar_trim():
    ybar_trim = np.array([[[3.58488052e-36, 6.32830768e-27, 2.09790680e-18, 4.40858955e-11,
         9.88649848e-11],
        [9.95643145e-22, 1.22425183e-12, 6.16311005e-11, 1.20920596e-07,
         1.03715742e-06],
        [1.44070685e-16, 8.83338101e-08, 2.64672512e-06, 1.09808357e-07,
         2.19529903e-06],
        [6.03573711e-18, 8.02163486e-09, 3.31869488e-06, 3.23194452e-06,
         7.56157633e-08],
        [2.51242987e-26, 7.57168685e-16, 1.89083309e-11, 3.37298279e-11,
         1.06578195e-14]],

       [[4.16242827e-36, 7.34783953e-27, 2.43589334e-18, 5.17079477e-11,
         1.15957846e-10],
        [1.15604778e-21, 1.42148682e-12, 7.15602748e-11, 1.41826673e-07,
         1.21672638e-06],
        [1.67281417e-16, 1.02564966e-07, 3.07312989e-06, 1.28842868e-07,
         2.57583877e-06],
        [7.03849384e-18, 9.37111939e-09, 3.88703574e-06, 3.79058205e-06,
         8.87101395e-08],
        [2.93596877e-26, 8.85973994e-16, 2.21527002e-11, 3.95484843e-11,
         1.25007824e-14]],

       [[2.88572878e-36, 5.09411108e-27, 1.68875642e-18, 3.87694679e-11,
         8.69426107e-11],
        [8.01464945e-22, 9.85488557e-13, 4.96113159e-11, 1.06338481e-07,
         8.99923917e-07],
        [1.15972881e-16, 7.11062519e-08, 2.13053986e-06, 9.23632983e-08,
         1.84653577e-06],
        [4.85809345e-18, 6.45521500e-09, 2.67329105e-06, 2.62893136e-06,
         6.21763748e-08],
        [2.02177444e-26, 6.09384952e-16, 1.52403018e-11, 2.73075542e-11,
         8.68313373e-15]],

       [[4.02295449e-36, 7.10162964e-27, 2.35427192e-18, 5.23788941e-11,
         1.17462479e-10],
        [1.11731117e-21, 1.37385594e-12, 6.91624479e-11, 1.43666972e-07,
         1.22783058e-06],
        [1.61676186e-16, 9.91282402e-08, 2.97015609e-06, 1.28705521e-07,
         2.57309291e-06],
        [6.80621648e-18, 9.07233408e-09, 3.78153210e-06, 3.72570633e-06,
         8.77558701e-08],
        [2.84320309e-26, 8.59650318e-16, 2.15760808e-11, 3.87246064e-11,
         1.23003167e-14]],

       [[3.87530796e-36, 6.84099261e-27, 2.26786775e-18, 5.09104520e-11,
         1.14169419e-10],
        [1.07630471e-21, 1.32343403e-12, 6.66241157e-11, 1.39639269e-07,
         1.19086423e-06],
        [1.55742505e-16, 9.54901329e-08, 2.86114833e-06, 1.24231276e-07,
         2.48364338e-06],
        [6.54646222e-18, 8.71955252e-09, 3.63055656e-06, 3.57912902e-06,
         8.44235798e-08],
        [2.73228794e-26, 8.25635437e-16, 2.07132533e-11, 3.71824445e-11,
         1.18186910e-14]],

       [[3.94587634e-36, 6.96556535e-27, 2.30916505e-18, 4.97038009e-11,
         1.11463439e-10],
        [1.09590395e-21, 1.34753342e-12, 6.78373239e-11, 1.36329617e-07,
         1.16508074e-06],
        [1.58578537e-16, 9.72289840e-08, 2.91324912e-06, 1.22265380e-07,
         2.44434100e-06],
        [6.63592237e-18, 8.81442298e-09, 3.64563870e-06, 3.56137125e-06,
         8.36134775e-08],
        [2.76049889e-26, 8.31690418e-16, 2.07743028e-11, 3.71103737e-11,
         1.17502715e-14]],

       [[3.65057522e-36, 6.44427703e-27, 2.13635197e-18, 4.55228128e-11,
         1.02087349e-10],
        [1.01388879e-21, 1.24668685e-12, 6.27605207e-11, 1.24861832e-07,
         1.06823720e-06],
        [1.46710851e-16, 8.99525705e-08, 2.69522766e-06, 1.12480145e-07,
         2.24871365e-06],
        [6.15390727e-18, 8.18029196e-09, 3.38346277e-06, 3.29568373e-06,
         7.71964686e-08],
        [2.56213905e-26, 7.72086559e-16, 1.92761263e-11, 3.43848757e-11,
         1.08695112e-14]]])

    return ybar_trim

def test_component_normalisation(my_stream_component):
    """Tests the normalisations of all densities evaluated for a component

    """
    cube, stream = my_stream_component
    ssps = cube.ssps
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
    cube, stream = my_stream_component
    ybar_trim = stream.ybar[::150, ::2, ::2]
    assert np.allclose(ybar_trim, my_stream_ybar_trim)


# end
