import pytest
import matplotlib.pyplot as plt
import numpy as np
import popkinmocks as pkm

@pytest.fixture
def my_stream_ybar_trim():
    ybar_trim = np.array(
        [[[3.47477823e-36, 6.13394662e-27, 2.03347387e-18, 4.29193180e-11,
         9.62488723e-11],
        [9.65063996e-22, 1.18665143e-12, 5.97382270e-11, 1.17720860e-07,
         1.00948149e-06],
        [1.39645848e-16, 8.56208172e-08, 2.56543636e-06, 1.06803770e-07,
         2.13523103e-06],
        [5.85197087e-18, 7.77903099e-09, 3.22029967e-06, 3.13930703e-06,
         7.34899029e-08],
        [2.43655448e-26, 7.34493669e-16, 1.83499981e-11, 3.27514134e-11,
         1.03533620e-14]],

       [[4.04040071e-36, 7.13242705e-27, 2.36448162e-18, 4.76254201e-11,
         1.06802558e-10],
        [1.12215658e-21, 1.37981389e-12, 6.94623825e-11, 1.30628949e-07,
         1.12244178e-06],
        [1.62377322e-16, 9.95581265e-08, 2.98303667e-06, 1.19571198e-07,
         2.39047867e-06],
        [6.78789411e-18, 9.00320422e-09, 3.70120407e-06, 3.56628733e-06,
         8.29595787e-08],
        [2.81859479e-26, 8.47169160e-16, 2.10600904e-11, 3.73566044e-11,
         1.17483378e-14]],

       [[3.58999746e-36, 6.33734050e-27, 2.10090128e-18, 4.35388138e-11,
         9.76381248e-11],
        [9.97064292e-22, 1.22599929e-12, 6.17190707e-11, 1.19420038e-07,
         1.02163084e-06],
        [1.44276327e-16, 8.84598948e-08, 2.65050297e-06, 1.07652384e-07,
         2.15219661e-06],
        [6.00850674e-18, 7.95844186e-09, 3.26981771e-06, 3.16639571e-06,
         7.40198118e-08],
        [2.49103344e-26, 7.48278303e-16, 1.86108986e-11, 3.30896971e-11,
         1.04389430e-14]],

       [[3.78744372e-36, 6.68588786e-27, 2.21644874e-18, 4.97776197e-11,
         1.11628982e-10],
        [1.05190183e-21, 1.29342802e-12, 6.51135576e-11, 1.36532090e-07,
         1.16492593e-06],
        [1.52211380e-16, 9.33251002e-08, 2.79627797e-06, 1.21536792e-07,
         2.42977501e-06],
        [6.38608076e-18, 8.50079691e-09, 3.53886016e-06, 3.49412784e-06,
         8.25047614e-08],
        [2.66353252e-26, 8.04691140e-16, 2.01924721e-11, 3.62759761e-11,
         1.15401640e-14]],

       [[3.96946737e-36, 7.00721006e-27, 2.32297075e-18, 5.06784618e-11,
         1.13649168e-10],
        [1.10245598e-21, 1.35558986e-12, 6.82428999e-11, 1.39002956e-07,
         1.18751146e-06],
        [1.59526623e-16, 9.78102826e-08, 2.93066643e-06, 1.24487770e-07,
         2.48877124e-06],
        [6.69250109e-18, 8.90161800e-09, 3.69194304e-06, 3.61706631e-06,
         8.50205935e-08],
        [2.78851551e-26, 8.41215056e-16, 2.10473226e-11, 3.76581055e-11,
         1.19366847e-14]],

       [[3.92736266e-36, 6.93288358e-27, 2.29833066e-18, 5.04963296e-11,
         1.13240727e-10],
        [1.09076207e-21, 1.34121093e-12, 6.75190377e-11, 1.38503397e-07,
         1.18350928e-06],
        [1.57834501e-16, 9.67727949e-08, 2.89958043e-06, 1.24097062e-07,
         2.48096017e-06],
        [6.63122442e-18, 8.82832745e-09, 3.66919702e-06, 3.60222255e-06,
         8.47245691e-08],
        [2.76606402e-26, 8.35243756e-16, 2.09245911e-11, 3.74822596e-11,
         1.18892889e-14]],

       [[3.68495827e-36, 6.50497265e-27, 2.15647327e-18, 4.61727791e-11,
         1.03544933e-10],
        [1.02343814e-21, 1.25842880e-12, 6.33516326e-11, 1.26644586e-07,
         1.08235411e-06],
        [1.48092652e-16, 9.07997915e-08, 2.72061274e-06, 1.13668530e-07,
         2.27247196e-06],
        [6.20034000e-18, 8.23626899e-09, 3.40453527e-06, 3.31952155e-06,
         7.78441843e-08],
        [2.57940202e-26, 7.76988144e-16, 1.93966547e-11, 3.46153033e-11,
         1.09500542e-14]]])

    return ybar_trim

def test_component_normalisation(my_stream_component):
    """Tests the normalisations of all densities evaluated for a component

    """
    ssps, cube, stream = my_stream_component
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
    my_stream_component,
    my_stream_ybar_trim,
    ):
    """Checks value of ybar for one component

    """
    ssps, cube, stream = my_stream_component
    ybar_trim = stream.ybar[::150, ::2, ::2]
    assert np.allclose(ybar_trim, my_stream_ybar_trim)





# end
