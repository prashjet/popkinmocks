import pytest
import matplotlib.pyplot as plt
import numpy as np
import popkinmocks as pkm

@pytest.fixture
def my_kinematic_maps():
    delta_E_v_x = np.array([[ 3.96427704e+00,  5.66168106e+00,  8.02208047e+00,
             1.03471035e+01,  9.30166732e+00,  9.30166732e+00,
             1.03471035e+01,  8.02208047e+00,  5.66168106e+00,
             3.96427704e+00],
           [ 3.31372057e+00,  4.95184032e+00,  7.57009012e+00,
             1.11179232e+01,  1.20348865e+01,  1.20348865e+01,
             1.11179232e+01,  7.57009012e+00,  4.95184032e+00,
             3.31372057e+00],
           [ 2.39856919e+00,  3.72226855e+00,  6.09640957e+00,
             1.02690388e+01,  1.42136979e+01,  1.42136979e+01,
             1.02690388e+01,  6.09640957e+00,  3.72226855e+00,
             2.39856919e+00],
           [ 1.26270953e+00,  2.00999807e+00,  3.46063797e+00,
             6.55726142e+00,  1.25417634e+01,  1.25417634e+01,
             6.55726142e+00,  3.46063797e+00,  2.00999807e+00,
             1.26270953e+00],
           [-4.76551143e-16, -5.83019663e-16, -7.05040171e-16,
            -7.83305734e-16, -5.00149584e-16, -5.00149584e-16,
            -7.83305734e-16, -7.05040171e-16, -5.83019663e-16,
            -4.76551143e-16],
           [-1.26270953e+00, -2.00999807e+00, -3.46063797e+00,
            -6.55726142e+00, -1.25417634e+01, -1.25417634e+01,
            -6.55726142e+00, -3.46063797e+00, -2.00999807e+00,
            -1.26270953e+00],
           [-2.39856919e+00, -3.72226855e+00, -6.09640957e+00,
            -1.02690388e+01, -1.42136979e+01, -1.42136979e+01,
            -1.02690388e+01, -6.09640957e+00, -3.72226855e+00,
            -2.39856919e+00],
           [-3.31372057e+00, -4.95184032e+00, -7.57009012e+00,
            -1.11179232e+01, -1.20348865e+01, -1.20348865e+01,
            -1.11179232e+01, -7.57009012e+00, -4.95184032e+00,
            -3.31372057e+00],
           [-3.96427704e+00, -5.66168106e+00, -8.02208047e+00,
            -1.03471035e+01, -9.30166732e+00, -9.30166732e+00,
            -1.03471035e+01, -8.02208047e+00, -5.66168106e+00,
            -3.96427704e+00]])

    delta_var_v_x = np.array(
        [[ -369.98852742,  -499.26015381,  -747.71160638, -1066.71800478,
             -889.82617473,  -889.82617473, -1066.71800478,  -747.71160638,
             -499.26015381,  -369.98852742],
           [ -337.77944403,  -447.44644323,  -703.33954597, -1211.20207506,
            -1359.64269146, -1359.64269146, -1211.20207506,  -703.33954597,
             -447.44644323,  -337.77944403],
           [ -300.90741248,  -369.99690954,  -555.02444516, -1084.27936653,
            -1826.62979344, -1826.62979344, -1084.27936653,  -555.02444516,
             -369.99690954,  -300.90741248],
           [ -271.06363851,  -297.16431161,  -363.5236072 ,  -602.37032274,
            -1483.64877835, -1483.64877835,  -602.37032274,  -363.5236072 ,
             -297.16431161,  -271.06363851],
           [ -259.53960045,  -266.93389706,  -271.51615845,  -265.32231284,
             -229.45055431,  -229.45055431,  -265.32231284,  -271.51615845,
             -266.93389706,  -259.53960045],
           [ -271.06363851,  -297.16431161,  -363.5236072 ,  -602.37032274,
            -1483.64877835, -1483.64877835,  -602.37032274,  -363.5236072 ,
             -297.16431161,  -271.06363851],
           [ -300.90741248,  -369.99690954,  -555.02444516, -1084.27936653,
            -1826.62979344, -1826.62979344, -1084.27936653,  -555.02444516,
             -369.99690954,  -300.90741248],
           [ -337.77944403,  -447.44644323,  -703.33954597, -1211.20207506,
            -1359.64269146, -1359.64269146, -1211.20207506,  -703.33954597,
             -447.44644323,  -337.77944403],
           [ -369.98852742,  -499.26015381,  -747.71160638, -1066.71800478,
             -889.82617473,  -889.82617473, -1066.71800478,  -747.71160638,
             -499.26015381,  -369.98852742]])

    delta_skew_v_x = np.array(
        [[ 2.20464475e-01,  2.91296455e-01,  3.49330106e-01,
             3.53901364e-01,  3.22743438e-01,  3.22743438e-01,
             3.53901364e-01,  3.49330106e-01,  2.91296455e-01,
             2.20464475e-01],
           [ 1.90979173e-01,  2.69756963e-01,  3.53596585e-01,
             3.82666676e-01,  3.59885764e-01,  3.59885764e-01,
             3.82666676e-01,  3.53596585e-01,  2.69756963e-01,
             1.90979173e-01],
           [ 1.43225907e-01,  2.17079893e-01,  3.21915230e-01,
             4.02454013e-01,  3.86696148e-01,  3.86696148e-01,
             4.02454013e-01,  3.21915230e-01,  2.17079893e-01,
             1.43225907e-01],
           [ 7.74597073e-02,  1.24407732e-01,  2.10750176e-01,
             3.49994032e-01,  4.24625243e-01,  4.24625243e-01,
             3.49994032e-01,  2.10750176e-01,  1.24407732e-01,
             7.74597073e-02],
           [-2.90236927e-17, -3.63435624e-17, -4.52797245e-17,
            -5.21826959e-17, -3.47056800e-17, -3.47056800e-17,
            -5.21826959e-17, -4.52797245e-17, -3.63435624e-17,
            -2.90236927e-17],
           [-7.74597073e-02, -1.24407732e-01, -2.10750176e-01,
            -3.49994032e-01, -4.24625243e-01, -4.24625243e-01,
            -3.49994032e-01, -2.10750176e-01, -1.24407732e-01,
            -7.74597073e-02],
           [-1.43225907e-01, -2.17079893e-01, -3.21915230e-01,
            -4.02454013e-01, -3.86696148e-01, -3.86696148e-01,
            -4.02454013e-01, -3.21915230e-01, -2.17079893e-01,
            -1.43225907e-01],
           [-1.90979173e-01, -2.69756963e-01, -3.53596585e-01,
            -3.82666676e-01, -3.59885764e-01, -3.59885764e-01,
            -3.82666676e-01, -3.53596585e-01, -2.69756963e-01,
            -1.90979173e-01],
           [-2.20464475e-01, -2.91296455e-01, -3.49330106e-01,
            -3.53901364e-01, -3.22743438e-01, -3.22743438e-01,
            -3.53901364e-01, -3.49330106e-01, -2.91296455e-01,
            -2.20464475e-01]])

    delta_kurt_v_x = np.array(
        [[-0.08075301, -0.09308142, -0.07506821, -0.01586614, -0.01462656,
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
    ybar = np.array([[[2.05669141e-07, 2.78728231e-07, 4.42344038e-07, 3.34926818e-07,
             2.37194483e-07],
            [2.12515730e-07, 2.98331821e-07, 4.98433563e-07, 3.69348139e-07,
             2.48331512e-07],
            [2.14811762e-07, 3.06092247e-07, 5.48102549e-07, 3.87787159e-07,
             2.52296764e-07],
            [2.11788928e-07, 2.95090711e-07, 4.73175372e-07, 3.61839018e-07,
             2.46859559e-07],
            [2.04443547e-07, 2.74074064e-07, 4.12026246e-07, 3.24713154e-07,
             2.34872437e-07]],

           [[2.24243590e-07, 3.07205199e-07, 5.01286594e-07, 3.73743468e-07,
             2.59582406e-07],
            [2.31198345e-07, 3.27377398e-07, 5.67715103e-07, 4.11231541e-07,
             2.70890333e-07],
            [2.33223042e-07, 3.33633112e-07, 6.14432639e-07, 4.25706675e-07,
             2.74258746e-07],
            [2.29619986e-07, 3.20175498e-07, 5.14745380e-07, 3.92305721e-07,
             2.67727668e-07],
            [2.21463494e-07, 2.96592314e-07, 4.50732588e-07, 3.50939694e-07,
             2.54353707e-07]],

           [[2.01327961e-07, 2.75214448e-07, 4.54420440e-07, 3.35612366e-07,
             2.32747862e-07],
            [2.07675489e-07, 2.92822375e-07, 5.11694058e-07, 3.67227660e-07,
             2.42905021e-07],
            [2.09731622e-07, 2.98921889e-07, 5.38213014e-07, 3.79229564e-07,
             2.46320236e-07],
            [2.06849639e-07, 2.88939298e-07, 4.77573965e-07, 3.56158683e-07,
             2.41242480e-07],
            [1.99837077e-07, 2.69087402e-07, 4.20915658e-07, 3.21370979e-07,
             2.29877244e-07]],

           [[2.11868284e-07, 2.87464026e-07, 4.67407801e-07, 3.48208996e-07,
             2.44228824e-07],
            [2.19133194e-07, 3.07617495e-07, 5.30466662e-07, 3.83869821e-07,
             2.55912059e-07],
            [2.22121390e-07, 3.17610660e-07, 5.85254161e-07, 4.05498506e-07,
             2.61123585e-07],
            [2.19995079e-07, 3.11406244e-07, 5.44159989e-07, 3.92646608e-07,
             2.57626373e-07],
            [2.13360139e-07, 2.92668227e-07, 4.81370286e-07, 3.57651504e-07,
             2.46965198e-07]],

           [[2.19783878e-07, 2.99777879e-07, 4.87566291e-07, 3.64394429e-07,
             2.53870492e-07],
            [2.27011075e-07, 3.20254291e-07, 5.55234472e-07, 4.02005542e-07,
             2.65551568e-07],
            [2.29675156e-07, 3.28930644e-07, 6.13393511e-07, 4.21317499e-07,
             2.70125173e-07],
            [2.26999728e-07, 3.20152126e-07, 5.54340503e-07, 4.01467433e-07,
             2.65524587e-07],
            [2.19752986e-07, 2.99504805e-07, 4.85738565e-07, 3.63545733e-07,
             2.53787142e-07]],

           [[2.17949187e-07, 2.96764405e-07, 4.78730957e-07, 3.59681987e-07,
             2.51631321e-07],
            [2.25210130e-07, 3.17626560e-07, 5.48237892e-07, 3.98122142e-07,
             2.63442450e-07],
            [2.27926559e-07, 3.26784477e-07, 6.13346376e-07, 4.19421684e-07,
             2.68157131e-07],
            [2.25300341e-07, 3.18049672e-07, 5.51657358e-07, 3.99331302e-07,
             2.63623971e-07],
            [2.18112533e-07, 2.97437207e-07, 4.82199737e-07, 3.61244217e-07,
             2.51946202e-07]],

           [[2.06593057e-07, 2.82235000e-07, 4.57412468e-07, 3.43163900e-07,
             2.38804471e-07],
            [2.13202144e-07, 3.01257924e-07, 5.23225461e-07, 3.78708517e-07,
             2.49530873e-07],
            [2.15406545e-07, 3.08229796e-07, 5.71885707e-07, 3.94449873e-07,
             2.53251968e-07],
            [2.12553383e-07, 2.98442634e-07, 5.09715665e-07, 3.72223642e-07,
             2.48244664e-07],
            [2.05474653e-07, 2.78425493e-07, 4.45986247e-07, 3.36229905e-07,
             2.36771020e-07]]])

    return ybar


def test_component_normalisation(my_component):
    """Tests the normalisations of all densities evaluated for a component

    """
    ssps, cube, gc1 = my_component
    v_edg = np.linspace(-900, 900, 20)
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
    # check p_tz_x
    a = gc1.get_p_tz_x(density=False, light_weighted=False)
    assert np.allclose(np.sum(a, (0,1)), 1.)
    a = gc1.get_p_tz_x(density=True, light_weighted=False)
    vol_elmt = ssps.delta_t[:,na,na,na]*ssps.delta_z[na,:,na,na]
    assert np.allclose(np.sum(a*vol_elmt, (0,1)), 1.)
    a = gc1.get_p_tz_x(density=False, light_weighted=True)
    assert np.allclose(np.sum(a, (0,1)), 1.)
    a = gc1.get_p_tz_x(density=True, light_weighted=True)
    vol_elmt = ssps.delta_t[:,na,na,na]*ssps.delta_z[na,:,na,na]
    assert np.allclose(np.sum(a*vol_elmt, (0,1)), 1.)
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
    a = gc1.get_p_v_tx(v_edg, density=False, light_weighted=False)
    assert np.allclose(np.sum(a, 0), 1.)
    a = gc1.get_p_v_tx(v_edg, density=True, light_weighted=False)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    a = gc1.get_p_v_tx(v_edg, density=False, light_weighted=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = gc1.get_p_v_tx(v_edg, density=True, light_weighted=True)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    # check get_p_tvxz
    a = gc1.get_p_tvxz(v_edg, density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1)
    a = gc1.get_p_tvxz(v_edg, density=True, light_weighted=False)
    vol_elmt = ssps.delta_t[:,na,na,na,na]*ssps.delta_z[na,na,na,na,:]
    vol_elmt *= cube.dx * cube.dy * dv
    assert np.isclose(np.sum(a*vol_elmt), 1)
    a = gc1.get_p_tvxz(v_edg, density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1)
    a = gc1.get_p_tvxz(v_edg, density=True, light_weighted=True)
    vol_elmt = ssps.delta_t[:,na,na,na,na]*ssps.delta_z[na,na,na,na,:]
    vol_elmt *= cube.dx * cube.dy * dv
    assert np.isclose(np.sum(a*vol_elmt), 1)
    # check get_p_v_x
    a = gc1.get_p_v_x(v_edg, density=False, light_weighted=False)
    assert np.allclose(np.sum(a, 0), 1.)
    a = gc1.get_p_v_x(v_edg, density=True, light_weighted=False)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    a = gc1.get_p_v_x(v_edg, density=False, light_weighted=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = gc1.get_p_v_x(v_edg, density=True, light_weighted=True)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    # check get_p_v
    a = gc1.get_p_v(v_edg, density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1.)
    a = gc1.get_p_v(v_edg, density=True, light_weighted=False)
    assert np.isclose(np.sum(a*dv), 1.)
    a = gc1.get_p_v(v_edg, density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1.)
    a = gc1.get_p_v(v_edg, density=True, light_weighted=True)
    assert np.isclose(np.sum(a*dv), 1.)



def test_component_kinematic_maps(
    my_component,
    my_kinematic_maps,
    ):
    """Checks Delta(mass_weighted kinematic maps- light weighted kinematic maps)

    """
    ssps, cube, gc1 = my_component
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
    ssps, cube, gc1 = my_component
    ybar_trim = gc1.ybar[::150, ::2, ::2]
    assert np.allclose(ybar_trim, my_ybar)







# end
