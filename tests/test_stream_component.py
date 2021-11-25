import pytest
import matplotlib.pyplot as plt
import numpy as np
import popkinmocks as pkm


@pytest.fixture
def my_component():
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

@pytest.fixture
def my_ybar_trim():
    ybar_trim = np.array([[
        [3.03578561e-19, 1.21683510e-12, 1.48492461e-10, 7.22051417e-12,
         1.69923785e-17],
        [1.80777789e-11, 4.07731651e-07, 1.72706117e-06, 9.16653014e-07,
         2.61568387e-10],
        [3.12313637e-09, 1.52308963e-06, 1.68466131e-06, 2.21762247e-06,
         6.07468918e-07],
        [1.64410263e-10, 9.36188998e-07, 1.04246152e-06, 9.56011383e-07,
         4.54685137e-07],
        [2.14724729e-16, 3.44183442e-10, 4.04382764e-08, 2.40623603e-08,
         4.85536254e-11]],

       [[3.52994911e-19, 1.41491084e-12, 1.72663981e-10, 8.24352748e-12,
         1.94399626e-17],
        [2.10204697e-11, 4.74101984e-07, 2.00819123e-06, 1.04822726e-06,
         2.99841925e-10],
        [3.63151878e-09, 1.77101731e-06, 1.95888954e-06, 2.55021156e-06,
         6.98574386e-07],
        [1.91019291e-10, 1.08704309e-06, 1.20827203e-06, 1.10464820e-06,
         5.24392893e-07],
        [2.49291938e-16, 3.99287710e-10, 4.68530950e-08, 2.78343789e-08,
         5.60846394e-11]],

       [[3.13644841e-19, 1.25718380e-12, 1.53416282e-10, 7.29859761e-12,
         1.71964159e-17],
        [1.86772151e-11, 4.21251515e-07, 1.78432833e-06, 9.27413235e-07,
         2.65056603e-10],
        [3.22669561e-09, 1.57359334e-06, 1.74052254e-06, 2.25278895e-06,
         6.17102001e-07],
        [1.69425721e-10, 9.63290602e-07, 1.06891019e-06, 9.75949882e-07,
         4.63177641e-07],
        [2.20875346e-16, 3.53487656e-10, 4.14392664e-08, 2.45993567e-08,
         4.95471136e-11]],

       [[3.30894992e-19, 1.32632765e-12, 1.61854023e-10, 7.98905855e-12,
         1.87588441e-17],
        [1.97044431e-11, 4.44419925e-07, 1.88246459e-06, 1.01241830e-06,
         2.88168361e-10],
        [3.40416063e-09, 1.66013940e-06, 1.83624953e-06, 2.43575915e-06,
         6.67222664e-07],
        [1.79263073e-10, 1.02106398e-06, 1.13811519e-06, 1.04592734e-06,
         4.98173934e-07],
        [2.34207377e-16, 3.75566196e-10, 4.41588053e-08, 2.63044557e-08,
         5.31329258e-11]],

       [[3.46797728e-19, 1.39007064e-12, 1.69632690e-10, 8.29874421e-12,
         1.95129275e-17],
        [2.06514340e-11, 4.65778642e-07, 1.97293540e-06, 1.05281004e-06,
         3.00141474e-10],
        [3.56776378e-09, 1.73992531e-06, 1.92449925e-06, 2.54197812e-06,
         6.96319015e-07],
        [1.87910545e-10, 1.07028364e-06, 1.19244906e-06, 1.09441726e-06,
         5.20761961e-07],
        [2.45492804e-16, 3.93601808e-10, 4.62609984e-08, 2.75383831e-08,
         5.55873843e-11]],

       [[3.43119194e-19, 1.37532597e-12, 1.67833371e-10, 8.25873311e-12,
         1.94117643e-17],
        [2.04323813e-11, 4.60838061e-07, 1.95200819e-06, 1.04743686e-06,
         2.98467817e-10],
        [3.52992000e-09, 1.72146967e-06, 1.90408581e-06, 2.52597302e-06,
         6.91934771e-07],
        [1.86024878e-10, 1.05992404e-06, 1.18193488e-06, 1.08603669e-06,
         5.17078047e-07],
        [2.43134057e-16, 3.89970946e-10, 4.58602427e-08, 2.73169616e-08,
         5.51675299e-11]],

       [[3.21941216e-19, 1.29043820e-12, 1.57474372e-10, 7.63109502e-12,
         1.79583083e-17],
        [1.91712554e-11, 4.32394248e-07, 1.83152648e-06, 9.68754748e-07,
         2.76460858e-10],
        [3.31204654e-09, 1.61521724e-06, 1.78656196e-06, 2.34467355e-06,
         6.42271767e-07],
        [1.74302064e-10, 9.92288743e-07, 1.10423750e-06, 1.01174816e-06,
         4.80982656e-07],
        [2.27580807e-16, 3.64691199e-10, 4.28295437e-08, 2.54728420e-08,
         5.13802296e-11]]])

    return ybar_trim

def test_component_normalisation(my_component):
    """Tests the normalisations of all densities evaluated for a component

    """
    ssps, cube, stream = my_component
    v_edg = np.linspace(-900, 900, 20)
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
    # check p_z_t
    a = stream.get_p_z_t(density=False, light_weighted=False)
    assert np.allclose(np.sum(a, 0), 1.)
    a = stream.get_p_z_t(density=True, light_weighted=False)
    assert np.allclose(np.sum(a*ssps.delta_z[:,na], 0), 1.)
    a = stream.get_p_z_t(density=False, light_weighted=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = stream.get_p_z_t(density=True, light_weighted=True)
    assert np.allclose(np.sum(a*ssps.delta_z[:,na], 0), 1.)
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
    # check p_tz_x
    a = stream.get_p_tz_x(density=False, light_weighted=False)
    assert np.allclose(np.sum(a, (0,1)), 1.)
    a = stream.get_p_tz_x(density=True, light_weighted=False)
    vol_elmt = ssps.delta_t[:,na,na,na]*ssps.delta_z[na,:,na,na]
    assert np.allclose(np.sum(a*vol_elmt, (0,1)), 1.)
    a = stream.get_p_tz_x(density=False, light_weighted=True)
    assert np.allclose(np.sum(a, (0,1)), 1.)
    a = stream.get_p_tz_x(density=True, light_weighted=True)
    vol_elmt = ssps.delta_t[:,na,na,na]*ssps.delta_z[na,:,na,na]
    assert np.allclose(np.sum(a*vol_elmt, (0,1)), 1.)
    # check p_tz
    a = stream.get_p_tz(density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1.)
    a = stream.get_p_tz(density=True, light_weighted=False)
    assert np.isclose(np.sum(a*ssps.delta_t[:,na]*ssps.delta_z[na,:]), 1.)
    a = stream.get_p_tz(density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1.)
    a = stream.get_p_tz(density=True, light_weighted=True)
    assert np.isclose(np.sum(a*ssps.delta_t[:,na]*ssps.delta_z[na,:]), 1.)
    # check get_p_v_x
    a = stream.get_p_v_x(v_edg, density=False, light_weighted=False)
    assert np.allclose(np.sum(a, 0), 1.)
    a = stream.get_p_v_x(v_edg, density=True, light_weighted=False)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    a = stream.get_p_v_x(v_edg, density=False, light_weighted=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = stream.get_p_v_x(v_edg, density=True, light_weighted=True)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    # check get_p_v_tx
    a = stream.get_p_v_tx(v_edg, density=False, light_weighted=False)
    assert np.allclose(np.sum(a, 0), 1.)
    a = stream.get_p_v_tx(v_edg, density=True, light_weighted=False)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    a = stream.get_p_v_tx(v_edg, density=False, light_weighted=True)
    assert np.allclose(np.sum(a, 0), 1.)
    a = stream.get_p_v_tx(v_edg, density=True, light_weighted=True)
    assert np.allclose(np.sum(a*dv, 0), 1.)
    # check get_p_v
    a = stream.get_p_v(v_edg, density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1.)
    a = stream.get_p_v(v_edg, density=True, light_weighted=False)
    assert np.isclose(np.sum(a*dv), 1.)
    a = stream.get_p_v(v_edg, density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1.)
    a = stream.get_p_v(v_edg, density=True, light_weighted=True)
    assert np.isclose(np.sum(a*dv), 1.)
    # check get_p_tvxz
    a = stream.get_p_tvxz(v_edg, density=False, light_weighted=False)
    assert np.isclose(np.sum(a), 1)
    a = stream.get_p_tvxz(v_edg, density=True, light_weighted=False)
    vol_elmt = ssps.delta_t[:,na,na,na,na]*ssps.delta_z[na,na,na,na,:]
    vol_elmt *= cube.dx * cube.dy * dv
    assert np.isclose(np.sum(a*vol_elmt), 1)
    a = stream.get_p_tvxz(v_edg, density=False, light_weighted=True)
    assert np.isclose(np.sum(a), 1)
    a = stream.get_p_tvxz(v_edg, density=True, light_weighted=True)
    vol_elmt = ssps.delta_t[:,na,na,na,na]*ssps.delta_z[na,na,na,na,:]
    vol_elmt *= cube.dx * cube.dy * dv
    assert np.isclose(np.sum(a*vol_elmt), 1)

def test_component_ybar(
    my_component,
    my_ybar_trim,
    ):
    """Checks value of ybar for one component

    """
    ssps, cube, stream = my_component
    ybar_trim = stream.ybar[::150, ::2, ::2]
    assert np.allclose(ybar_trim, my_ybar_trim)





# end
