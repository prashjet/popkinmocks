import pytest
import matplotlib.pyplot as plt
import numpy as np
import popkinmocks as pkm

@pytest.fixture
def my_kinematic_maps():
    delta_E_v_x = np.array([[ 3.96427705e+00,  5.66168107e+00,  8.02208048e+00,
         1.03471035e+01,  9.30166733e+00,  9.30166733e+00,
         1.03471035e+01,  8.02208048e+00,  5.66168107e+00,
         3.96427705e+00],
       [ 3.31372058e+00,  4.95184033e+00,  7.57009013e+00,
         1.11179232e+01,  1.20348865e+01,  1.20348865e+01,
         1.11179232e+01,  7.57009013e+00,  4.95184033e+00,
         3.31372058e+00],
       [ 2.39856920e+00,  3.72226855e+00,  6.09640958e+00,
         1.02690388e+01,  1.42136979e+01,  1.42136979e+01,
         1.02690388e+01,  6.09640958e+00,  3.72226855e+00,
         2.39856920e+00],
       [ 1.26270953e+00,  2.00999807e+00,  3.46063797e+00,
         6.55726143e+00,  1.25417634e+01,  1.25417634e+01,
         6.55726143e+00,  3.46063797e+00,  2.00999807e+00,
         1.26270953e+00],
       [-4.76551144e-16, -5.83019664e-16, -7.05040172e-16,
        -7.83305735e-16, -5.00149585e-16, -5.00149585e-16,
        -7.83305735e-16, -7.05040172e-16, -5.83019664e-16,
        -4.76551144e-16],
       [-1.26270953e+00, -2.00999807e+00, -3.46063797e+00,
        -6.55726143e+00, -1.25417634e+01, -1.25417634e+01,
        -6.55726143e+00, -3.46063797e+00, -2.00999807e+00,
        -1.26270953e+00],
       [-2.39856920e+00, -3.72226855e+00, -6.09640958e+00,
        -1.02690388e+01, -1.42136979e+01, -1.42136979e+01,
        -1.02690388e+01, -6.09640958e+00, -3.72226855e+00,
        -2.39856920e+00],
       [-3.31372058e+00, -4.95184033e+00, -7.57009013e+00,
        -1.11179232e+01, -1.20348865e+01, -1.20348865e+01,
        -1.11179232e+01, -7.57009013e+00, -4.95184033e+00,
        -3.31372058e+00],
       [-3.96427705e+00, -5.66168107e+00, -8.02208048e+00,
        -1.03471035e+01, -9.30166733e+00, -9.30166733e+00,
        -1.03471035e+01, -8.02208048e+00, -5.66168107e+00,
        -3.96427705e+00]])

    delta_var_v_x = np.array([[ -369.98852797,  -499.26015456,  -747.71160748, -1066.7180062 ,
         -889.82617578,  -889.82617578, -1066.7180062 ,  -747.71160748,
         -499.26015456,  -369.98852797],
       [ -337.77944452,  -447.4464439 ,  -703.33954705, -1211.20207687,
        -1359.64269333, -1359.64269333, -1211.20207687,  -703.33954705,
         -447.4464439 ,  -337.77944452],
       [ -300.90741291,  -369.99691009,  -555.02444603, -1084.27936828,
        -1826.62979629, -1826.62979629, -1084.27936828,  -555.02444603,
         -369.99691009,  -300.90741291],
       [ -271.06363888,  -297.16431203,  -363.52360775,  -602.37032372,
        -1483.64878087, -1483.64878087,  -602.37032372,  -363.52360775,
         -297.16431203,  -271.06363888],
       [ -259.5396008 ,  -266.93389743,  -271.51615883,  -265.32231322,
         -229.45055463,  -229.45055463,  -265.32231322,  -271.51615883,
         -266.93389743,  -259.5396008 ],
       [ -271.06363888,  -297.16431203,  -363.52360775,  -602.37032372,
        -1483.64878087, -1483.64878087,  -602.37032372,  -363.52360775,
         -297.16431203,  -271.06363888],
       [ -300.90741291,  -369.99691009,  -555.02444603, -1084.27936828,
        -1826.62979629, -1826.62979629, -1084.27936828,  -555.02444603,
         -369.99691009,  -300.90741291],
       [ -337.77944452,  -447.4464439 ,  -703.33954705, -1211.20207687,
        -1359.64269333, -1359.64269333, -1211.20207687,  -703.33954705,
         -447.4464439 ,  -337.77944452],
       [ -369.98852797,  -499.26015456,  -747.71160748, -1066.7180062 ,
         -889.82617578,  -889.82617578, -1066.7180062 ,  -747.71160748,
         -499.26015456,  -369.98852797]])

    delta_skew_v_x = np.array([[ 2.20464476e-01,  2.91296456e-01,  3.49330107e-01,
         3.53901364e-01,  3.22743438e-01,  3.22743438e-01,
         3.53901364e-01,  3.49330107e-01,  2.91296456e-01,
         2.20464476e-01],
       [ 1.90979174e-01,  2.69756964e-01,  3.53596586e-01,
         3.82666676e-01,  3.59885764e-01,  3.59885764e-01,
         3.82666676e-01,  3.53596586e-01,  2.69756964e-01,
         1.90979174e-01],
       [ 1.43225907e-01,  2.17079894e-01,  3.21915230e-01,
         4.02454013e-01,  3.86696149e-01,  3.86696149e-01,
         4.02454013e-01,  3.21915230e-01,  2.17079894e-01,
         1.43225907e-01],
       [ 7.74597074e-02,  1.24407732e-01,  2.10750176e-01,
         3.49994033e-01,  4.24625243e-01,  4.24625243e-01,
         3.49994033e-01,  2.10750176e-01,  1.24407732e-01,
         7.74597074e-02],
       [-2.90236927e-17, -3.63435625e-17, -4.52797246e-17,
        -5.21826959e-17, -3.47056800e-17, -3.47056800e-17,
        -5.21826959e-17, -4.52797246e-17, -3.63435625e-17,
        -2.90236927e-17],
       [-7.74597074e-02, -1.24407732e-01, -2.10750176e-01,
        -3.49994033e-01, -4.24625243e-01, -4.24625243e-01,
        -3.49994033e-01, -2.10750176e-01, -1.24407732e-01,
        -7.74597074e-02],
       [-1.43225907e-01, -2.17079894e-01, -3.21915230e-01,
        -4.02454013e-01, -3.86696149e-01, -3.86696149e-01,
        -4.02454013e-01, -3.21915230e-01, -2.17079894e-01,
        -1.43225907e-01],
       [-1.90979174e-01, -2.69756964e-01, -3.53596586e-01,
        -3.82666676e-01, -3.59885764e-01, -3.59885764e-01,
        -3.82666676e-01, -3.53596586e-01, -2.69756964e-01,
        -1.90979174e-01],
       [-2.20464476e-01, -2.91296456e-01, -3.49330107e-01,
        -3.53901364e-01, -3.22743438e-01, -3.22743438e-01,
        -3.53901364e-01, -3.49330107e-01, -2.91296456e-01,
        -2.20464476e-01]])

    delta_kurt_v_x = np.array([[-0.08075301, -0.09308142, -0.07506821, -0.01586614, -0.01462656,
        -0.01462656, -0.01586614, -0.07506821, -0.09308142, -0.08075301],
       [-0.0774494 , -0.09698931, -0.09452485, -0.01482313,  0.02263566,
         0.02263566, -0.01482313, -0.09452485, -0.09698931, -0.0774494 ],
       [-0.06981545, -0.09043261, -0.11291762, -0.05264256,  0.06741845,
         0.06741845, -0.05264256, -0.11291762, -0.09043261, -0.06981545],
       [-0.06103225, -0.07327925, -0.09789219, -0.12551736,  0.01850009,
         0.01850009, -0.12551736, -0.09789219, -0.07327925, -0.06103225],
       [-0.05701946, -0.06273223, -0.06919628, -0.07348541, -0.06579786,
        -0.06579786, -0.07348541, -0.06919628, -0.06273223, -0.05701946],
       [-0.06103225, -0.07327925, -0.09789219, -0.12551736,  0.01850009,
         0.01850009, -0.12551736, -0.09789219, -0.07327925, -0.06103225],
       [-0.06981545, -0.09043261, -0.11291762, -0.05264256,  0.06741845,
         0.06741845, -0.05264256, -0.11291762, -0.09043261, -0.06981545],
       [-0.0774494 , -0.09698931, -0.09452485, -0.01482313,  0.02263566,
         0.02263566, -0.01482313, -0.09452485, -0.09698931, -0.0774494 ],
       [-0.08075301, -0.09308142, -0.07506821, -0.01586614, -0.01462656,
        -0.01462656, -0.01586614, -0.07506821, -0.09308142, -0.08075301]])
    return delta_E_v_x, delta_var_v_x, delta_skew_v_x, delta_kurt_v_x


@pytest.fixture
def my_ybar():
    ybar = np.array([[[2.05505088e-07, 2.78581152e-07, 4.42244355e-07, 3.34815904e-07,
         2.37031109e-07],
        [2.12331010e-07, 2.98151266e-07, 4.98421088e-07, 3.69229986e-07,
         2.48138234e-07],
        [2.14603319e-07, 3.05815659e-07, 5.47842982e-07, 3.87488385e-07,
         2.52056828e-07],
        [2.11563395e-07, 2.94759249e-07, 4.72853582e-07, 3.61502395e-07,
         2.46587554e-07],
        [2.04214129e-07, 2.73762501e-07, 4.11691165e-07, 3.24390067e-07,
         2.34600070e-07]],

       [[2.28176558e-07, 3.11066817e-07, 4.98253378e-07, 3.76199037e-07,
         2.63654943e-07],
        [2.35588852e-07, 3.32451020e-07, 5.64561645e-07, 4.15648143e-07,
         2.75704277e-07],
        [2.38140005e-07, 3.40874408e-07, 6.31913682e-07, 4.35592405e-07,
         2.80084531e-07],
        [2.35030351e-07, 3.29860994e-07, 5.43464228e-07, 4.08577554e-07,
         2.74581578e-07],
        [2.27186906e-07, 3.07182378e-07, 4.78391712e-07, 3.67594989e-07,
         2.61777539e-07]],

       [[1.63720241e-07, 2.24152838e-07, 3.58620283e-07, 2.71871411e-07,
         1.89410262e-07],
        [1.68537459e-07, 2.37194355e-07, 3.96681163e-07, 2.96022851e-07,
         1.96940457e-07],
        [1.69964847e-07, 2.40305169e-07, 4.11202643e-07, 3.00695632e-07,
         1.99091562e-07],
        [1.67735568e-07, 2.34185835e-07, 4.21372309e-07, 2.92769368e-07,
         1.95395392e-07],
        [1.62398718e-07, 2.21128961e-07, 3.83648589e-07, 2.72034195e-07,
         1.87214111e-07]],

       [[2.21265303e-07, 3.00145399e-07, 4.80183862e-07, 3.62293179e-07,
         2.55114375e-07],
        [2.28837085e-07, 3.21609678e-07, 5.46122413e-07, 4.00941272e-07,
         2.67389692e-07],
        [2.31860574e-07, 3.32014310e-07, 6.18124212e-07, 4.24971215e-07,
         2.72699011e-07],
        [2.29439829e-07, 3.24185726e-07, 5.57723397e-07, 4.06594698e-07,
         2.68581658e-07],
        [2.22292366e-07, 3.03498155e-07, 4.89277892e-07, 3.68052882e-07,
         2.56958558e-07]],

       [[2.13909392e-07, 2.90444158e-07, 4.66450639e-07, 3.51243830e-07,
         2.46680242e-07],
        [2.21150488e-07, 3.10796379e-07, 5.30800972e-07, 3.88041380e-07,
         2.58363544e-07],
        [2.24008660e-07, 3.20298947e-07, 5.90895914e-07, 4.09102267e-07,
         2.63324505e-07],
        [2.21673182e-07, 3.13109724e-07, 5.48204153e-07, 3.93772929e-07,
         2.59404771e-07],
        [2.14816890e-07, 2.93727071e-07, 4.82078145e-07, 3.58056617e-07,
         2.48357262e-07]],

       [[2.11507144e-07, 2.89271432e-07, 4.75559439e-07, 3.52968210e-07,
         2.44530151e-07],
        [2.18306112e-07, 3.08477323e-07, 5.41536086e-07, 3.88389154e-07,
         2.55491791e-07],
        [2.20696356e-07, 3.15985642e-07, 5.87851183e-07, 4.04602453e-07,
         2.59536220e-07],
        [2.17986934e-07, 3.07038868e-07, 5.29440183e-07, 3.84574998e-07,
         2.54854871e-07],
        [2.10938935e-07, 2.87069124e-07, 4.64046366e-07, 3.48045746e-07,
         2.43457049e-07]],

       [[2.00089902e-07, 2.73460194e-07, 4.40166874e-07, 3.32126555e-07,
         2.31338121e-07],
        [2.06434586e-07, 2.91709878e-07, 5.01221000e-07, 3.66417408e-07,
         2.41618133e-07],
        [2.08514364e-07, 2.98201060e-07, 5.51145261e-07, 3.81199734e-07,
         2.45110485e-07],
        [2.05709296e-07, 2.88508077e-07, 4.86574299e-07, 3.58787202e-07,
         2.40176160e-07],
        [1.98823834e-07, 2.68949697e-07, 4.25937088e-07, 3.23607547e-07,
         2.29002349e-07]]])

    return ybar


def test_component_normalisation(my_component):
    """Tests the normalisations of all densities evaluated for a component

    """
    cube, gc1 = my_component
    ssps = cube.ssps
    v_edg = cube.v_edg
    dv = v_edg[1] - v_edg[0]
    na = np.newaxis
    # check p_t
    a = gc1.get_p_t(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1)
    a = gc1.get_p_t(density=True, light_weighted=False)
    assert np.isclose(np.sum(a*ssps.delta_t), 1)
    a = gc1.get_p_t(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1)
    a = gc1.get_p_t(density=True, light_weighted=True)
    assert np.isclose(np.sum(a*ssps.delta_t), 1)
    # check p_x_t
    a = gc1.get_p_x_t(density=False, light_weighted=False)
    assert np.allclose(np.sum(a, (0,1)), 1.)
    a = gc1.get_p_x_t(density=True, light_weighted=False)
    assert np.allclose(np.sum(a*cube.dx*cube.dy, (0,1)), 1.)
    a = gc1.get_p_x_t(density=False, light_weighted=True)
    assert np.allclose(np.sum(a, (0,1)), 1.)
    a = gc1.get_p_x_t(density=True, light_weighted=True)
    assert np.allclose(np.sum(a*cube.dx*cube.dy, (0,1)), 1.)
    # check p_txz
    a = gc1.get_p_txz(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1)
    a = gc1.get_p_txz(density=True, light_weighted=False)
    vol_elmt = ssps.delta_t[:,na,na,na]*cube.dx*cube.dy*ssps.delta_z[na,na,na,:]
    assert np.isclose(np.sum(a * vol_elmt), 1)
    a = gc1.get_p_txz(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1)
    a = gc1.get_p_txz(density=True, light_weighted=True)
    vol_elmt = ssps.delta_t[:,na,na,na]*cube.dx*cube.dy*ssps.delta_z[na,na,na,:]
    assert np.isclose(np.sum(a * vol_elmt), 1)
    # check p_tx
    a = gc1.get_p_tx(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1.)
    a = gc1.get_p_tx(density=True, light_weighted=False)
    assert np.isclose(np.sum(a*ssps.delta_t[:,na,na]*cube.dx*cube.dy), 1)
    a = gc1.get_p_tx(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1.)
    a = gc1.get_p_tx(density=True, light_weighted=True)
    assert np.isclose(np.sum(a*ssps.delta_t[:,na,na]*cube.dx*cube.dy), 1)
    # check p_z_tx
    a = gc1.get_p_z_tx(density=False, light_weighted=False)
    assert np.allclose(np.sum(a, 0), 1.)
    a = gc1.get_p_z_tx(density=True, light_weighted=False)
    assert np.allclose(np.sum(a*ssps.delta_z[:,na,na,na], 0), 1.)
    a = gc1.get_p_z_tx(density=False, light_weighted=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = gc1.get_p_z_tx(density=True, light_weighted=True)
    assert np.allclose(np.sum(a*ssps.delta_z[:,na,na,na], 0), 1.)
    # check get_p_x
    a = gc1.get_p_x(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1)
    a = gc1.get_p_x(density=True, light_weighted=False)
    assert np.isclose(np.sum(a*cube.dx*cube.dy), 1)
    a = gc1.get_p_x(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1)
    a = gc1.get_p_x(density=True, light_weighted=True)
    assert np.isclose(np.sum(a*cube.dx*cube.dy), 1)
    # check get_p_z
    a = gc1.get_p_z(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1)
    a = gc1.get_p_z(density=True, light_weighted=False)
    assert np.isclose(np.sum(a*ssps.delta_z), 1)
    a = gc1.get_p_z(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1)
    a = gc1.get_p_z(density=True, light_weighted=True)
    assert np.isclose(np.sum(a*ssps.delta_z), 1)
    # check p_tz
    a = gc1.get_p_tz(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1.)
    a = gc1.get_p_tz(density=True, light_weighted=False)
    assert np.isclose(np.sum(a*ssps.delta_t[:,na]*ssps.delta_z[na,:]), 1.)
    a = gc1.get_p_tz(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1.)
    a = gc1.get_p_tz(density=True, light_weighted=True)
    assert np.isclose(np.sum(a*ssps.delta_t[:,na]*ssps.delta_z[na,:]), 1.)
    # check get_p_v_tx
    a = gc1.get_p_v_tx(density=False, light_weighted=False)
    assert np.allclose(np.sum(a, 0), 1.)
    a = gc1.get_p_v_tx(density=True, light_weighted=False)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    a = gc1.get_p_v_tx(density=False, light_weighted=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = gc1.get_p_v_tx(density=True, light_weighted=True)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    # check get_p_tvxz
    a = gc1.get_p_tvxz(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1)
    a = gc1.get_p_tvxz(density=True, light_weighted=False)
    vol_elmt = ssps.delta_t[:,na,na,na,na]*ssps.delta_z[na,na,na,na,:]
    vol_elmt *= cube.dx * cube.dy * dv
    assert np.isclose(np.sum(a*vol_elmt), 1)
    a = gc1.get_p_tvxz(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1)
    a = gc1.get_p_tvxz(density=True, light_weighted=True)
    vol_elmt = ssps.delta_t[:,na,na,na,na]*ssps.delta_z[na,na,na,na,:]
    vol_elmt *= cube.dx * cube.dy * dv
    assert np.isclose(np.sum(a*vol_elmt), 1)
    # check get_p_v
    a = gc1.get_p_v(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1.)
    a = gc1.get_p_v(density=True, light_weighted=False)
    assert np.isclose(np.sum(a*dv), 1.)
    a = gc1.get_p_v(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1.)
    a = gc1.get_p_v(density=True, light_weighted=True)
    assert np.isclose(np.sum(a*dv), 1.)
    # check p_tz_x
    a = gc1.get_p('tz_x', density=False, light_weighted=False)
    assert np.allclose(np.sum(a, (0,1)), 1.)
    a = gc1.get_p('tz_x', density=True, light_weighted=False)
    vol_elmt = ssps.delta_t[:,na,na,na]*ssps.delta_z[na,:,na,na]
    assert np.allclose(np.sum(a*vol_elmt, (0,1)), 1.)
    a = gc1.get_p('tz_x', density=False, light_weighted=True)
    assert np.allclose(np.sum(a, (0,1)), 1.)
    a = gc1.get_p('tz_x', density=True, light_weighted=True)
    vol_elmt = ssps.delta_t[:,na,na,na]*ssps.delta_z[na,:,na,na]
    assert np.allclose(np.sum(a*vol_elmt, (0,1)), 1.)
    # check get_p_v_x
    a = gc1.get_p('v_x', density=False, light_weighted=False)
    assert np.allclose(np.sum(a, 0), 1.)
    a = gc1.get_p('v_x', density=True, light_weighted=False)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    a = gc1.get_p('v_x', density=False, light_weighted=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = gc1.get_p('v_x', density=True, light_weighted=True)
    assert np.allclose(np.sum(a*dv, 0), 1.)

def test_component_kinematic_maps(
    my_component,
    my_kinematic_maps,
    ):
    """Checks Delta(mass_weighted kinematic maps- light weighted kinematic maps)

    """
    cube, gc1 = my_component
    del_E_v_x, del_var_v_x, del_skew_v_x, del_kurt_v_x = my_kinematic_maps
    # check get_E_v_x
    a = gc1.get_E_v_x(light_weighted=False)
    b = gc1.get_E_v_x(light_weighted=True)
    assert np.allclose(a-b, del_E_v_x)
    # check get_variance_v_x
    a = gc1.get_variance_v_x(light_weighted=False)
    b = gc1.get_variance_v_x(light_weighted=True)
    assert np.allclose(a-b, del_var_v_x)
    # check get_skewness_v_x
    a = gc1.get_skewness_v_x(light_weighted=False)
    b = gc1.get_skewness_v_x(light_weighted=True)
    assert np.allclose(a-b, del_skew_v_x)
    # check get_kurtosis_v_x
    a = gc1.get_kurtosis_v_x(light_weighted=False)
    b = gc1.get_kurtosis_v_x(light_weighted=True)
    assert np.allclose(a-b, del_kurt_v_x)



def test_component_ybar(
    my_component,
    my_ybar,
    ):
    """Checks value of ybar for one component

    """
    cube, gc1 = my_component
    ybar_trim = gc1.ybar[::150, ::2, ::2]
    assert np.allclose(ybar_trim, my_ybar)







# end
