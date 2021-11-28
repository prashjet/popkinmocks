import pytest
import matplotlib.pyplot as plt
import numpy as np
import popkinmocks as pkm

@pytest.fixture
def my_kinematic_maps():
    delta_E_v_x = np.array([
            [-2.63231254, -2.91232087, -2.98508593, -1.98673529,  3.78808848,
             3.78808848, -1.98673529, -2.98508593, -2.91232087, -2.63231254],
            [-2.13502906, -2.45075383, -2.69793969, -2.22245885,  3.19109946,
             3.19109946, -2.22245885, -2.69793969, -2.45075383, -2.13502906],
            [-1.51721452, -1.80255015, -2.1309258 , -2.17208163,  1.81371172,
             1.81371172, -2.17208163, -2.1309258 , -1.80255015, -1.51721452],
            [-0.79161243, -0.96531765, -1.21070541, -1.49427586, -0.1610911 ,
             -0.1610911 , -1.49427586, -1.21070541, -0.96531765, -0.79161243],
            [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
            [ 0.79161243,  0.96531765,  1.21070541,  1.49427586,  0.1610911 ,
             0.1610911 ,  1.49427586,  1.21070541,  0.96531765,  0.79161243],
            [ 1.51721452,  1.80255015,  2.1309258 ,  2.17208163, -1.81371172,
             -1.81371172,  2.17208163,  2.1309258 ,  1.80255015,  1.51721452],
            [ 2.13502906,  2.45075383,  2.69793969,  2.22245885, -3.19109946,
             -3.19109946,  2.22245885,  2.69793969,  2.45075383,  2.13502906],
            [ 2.63231254,  2.91232087,  2.98508593,  1.98673529, -3.78808848,
             -3.78808848,  1.98673529,  2.98508593,  2.91232087,  2.63231254]])
    delta_var_v_x = np.array([
        [-280.21425525, -291.19986619, -291.65748034, -244.40588447,
         -293.61176514, -293.61176514, -244.40588447, -291.65748034,
         -291.19986619, -280.21425525],
        [-262.32619221, -272.2404863 , -278.53941553, -250.2600659 ,
         -257.36865076, -257.36865076, -250.2600659 , -278.53941553,
         -272.2404863 , -262.32619221],
        [-245.37601092, -251.26140478, -257.12608689, -246.94594179,
         -202.56176355, -202.56176355, -246.94594179, -257.12608689,
         -251.26140478, -245.37601092],
        [-232.85496802, -233.77421875, -233.61601769, -227.79587279,
         -181.7993307 , -181.7993307 , -227.79587279, -233.61601769,
         -233.77421875, -232.85496802],
        [-228.18799987, -226.78761345, -222.59783462, -211.84505418,
         -195.8310754 , -195.8310754 , -211.84505418, -222.59783462,
         -226.78761345, -228.18799987],
        [-232.85496802, -233.77421875, -233.61601769, -227.79587279,
         -181.7993307 , -181.7993307 , -227.79587279, -233.61601769,
         -233.77421875, -232.85496802],
        [-245.37601092, -251.26140478, -257.12608689, -246.94594179,
         -202.56176355, -202.56176355, -246.94594179, -257.12608689,
         -251.26140478, -245.37601092],
        [-262.32619221, -272.2404863 , -278.53941553, -250.2600659 ,
         -257.36865076, -257.36865076, -250.2600659 , -278.53941553,
         -272.2404863 , -262.32619221],
        [-280.21425525, -291.19986619, -291.65748034, -244.40588447,
         -293.61176514, -293.61176514, -244.40588447, -291.65748034,
         -291.19986619, -280.21425525]])
    delta_skew_v_x = np.array([
        [-0.14324404, -0.15862658, -0.16467698, -0.11463442,  0.18223122,
         0.18223122, -0.11463442, -0.16467698, -0.15862658, -0.14324404],
        [-0.1171153 , -0.13410869, -0.14820029, -0.1258044 ,  0.15951882,
         0.15951882, -0.1258044 , -0.14820029, -0.13410869, -0.1171153 ],
        [-0.08405618, -0.09957583, -0.11747619, -0.121175  ,  0.09641303,
         0.09641303, -0.121175  , -0.11747619, -0.09957583, -0.08405618],
        [-0.04424465, -0.05391409, -0.0675346 , -0.08357988, -0.009748  ,
         -0.009748  , -0.08357988, -0.0675346 , -0.05391409, -0.04424465],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
        [ 0.04424465,  0.05391409,  0.0675346 ,  0.08357988,  0.009748  ,
         0.009748  ,  0.08357988,  0.0675346 ,  0.05391409,  0.04424465],
        [ 0.08405618,  0.09957583,  0.11747619,  0.121175  , -0.09641303,
         -0.09641303,  0.121175  ,  0.11747619,  0.09957583,  0.08405618],
        [ 0.1171153 ,  0.13410869,  0.14820029,  0.1258044 , -0.15951882,
         -0.15951882,  0.1258044 ,  0.14820029,  0.13410869,  0.1171153 ],
        [ 0.14324404,  0.15862658,  0.16467698,  0.11463442, -0.18223122,
         -0.18223122,  0.11463442,  0.16467698,  0.15862658,  0.14324404]])
    delta_kurt_v_x = np.array([
        [-0.05086873, -0.05405702, -0.05568311, -0.04437223, -0.03793468,
         -0.03793468, -0.04437223, -0.05568311, -0.05405702, -0.05086873],
        [-0.04656647, -0.04921657, -0.05156603, -0.0461324 , -0.03620234,
         -0.03620234, -0.0461324 , -0.05156603, -0.04921657, -0.04656647],
        [-0.04239277, -0.04396597, -0.04576849, -0.04462888, -0.03099179,
         -0.03099179, -0.04462888, -0.04576849, -0.04396597, -0.04239277],
        [-0.03917829, -0.03946778, -0.03966134, -0.03926559, -0.03064167,
         -0.03064167, -0.03926559, -0.03966134, -0.03946778, -0.03917829],
        [-0.03794008, -0.03760283, -0.03672604, -0.03514706, -0.03743389,
         -0.03743389, -0.03514706, -0.03672604, -0.03760283, -0.03794008],
        [-0.03917829, -0.03946778, -0.03966134, -0.03926559, -0.03064167,
         -0.03064167, -0.03926559, -0.03966134, -0.03946778, -0.03917829],
        [-0.04239277, -0.04396597, -0.04576849, -0.04462888, -0.03099179,
         -0.03099179, -0.04462888, -0.04576849, -0.04396597, -0.04239277],
        [-0.04656647, -0.04921657, -0.05156603, -0.0461324 , -0.03620234,
         -0.03620234, -0.0461324 , -0.05156603, -0.04921657, -0.04656647],
        [-0.05086873, -0.05405702, -0.05568311, -0.04437223, -0.03793468,
         -0.03793468, -0.04437223, -0.05568311, -0.05405702, -0.05086873]])
    return delta_E_v_x, delta_var_v_x, delta_skew_v_x, delta_kurt_v_x


@pytest.fixture
def my_ybar():
    ybar = np.array([[
        [1.66503907e-07, 2.50790314e-07, 5.14674482e-07, 3.34792026e-07,
         2.00169632e-07],
        [1.72432910e-07, 2.73377303e-07, 6.41443532e-07, 3.87723843e-07,
         2.11101778e-07],
        [1.73808494e-07, 2.81770920e-07, 7.99925189e-07, 4.16908084e-07,
         2.14249485e-07],
        [1.69921633e-07, 2.67298718e-07, 6.14766128e-07, 3.76492694e-07,
         2.07368985e-07],
        [1.62462075e-07, 2.42726446e-07, 4.79021326e-07, 3.21478885e-07,
         1.94621366e-07]],
       [[1.82074881e-07, 2.76814846e-07, 5.81555277e-07, 3.73077668e-07,
         2.19687251e-07],
        [1.87718002e-07, 2.99395561e-07, 7.30464312e-07, 4.28388405e-07,
         2.30306513e-07],
        [1.88770248e-07, 3.06619783e-07, 8.85958564e-07, 4.54873360e-07,
         2.32873273e-07],
        [1.84696877e-07, 2.91174753e-07, 6.67850264e-07, 4.10589086e-07,
         2.25614454e-07],
        [1.76838721e-07, 2.64512976e-07, 5.25959705e-07, 3.50300927e-07,
         2.11999609e-07]],
       [[1.63627114e-07, 2.48525524e-07, 5.25187306e-07, 3.34836620e-07,
         1.97338003e-07],
        [1.68708135e-07, 2.68573140e-07, 6.52644330e-07, 3.83549190e-07,
         2.06820312e-07],
        [1.69921000e-07, 2.75567210e-07, 7.84323467e-07, 4.07850387e-07,
         2.09491566e-07],
        [1.66822897e-07, 2.63473972e-07, 6.13428884e-07, 3.72590034e-07,
         2.03899524e-07],
        [1.60325493e-07, 2.40759492e-07, 4.88627858e-07, 3.20425681e-07,
         1.92487762e-07]],
       [[1.70426217e-07, 2.57397205e-07, 5.38202276e-07, 3.45194536e-07,
         2.05046627e-07],
        [1.76854175e-07, 2.80221904e-07, 6.73254196e-07, 3.98333138e-07,
         2.16412619e-07],
        [1.79803094e-07, 2.91937009e-07, 8.44428120e-07, 4.32949595e-07,
         2.21778297e-07],
        [1.78412976e-07, 2.84434418e-07, 7.01377895e-07, 4.07341196e-07,
         2.18828490e-07],
        [1.73084470e-07, 2.63494388e-07, 5.58817773e-07, 3.56047725e-07,
         2.08916742e-07]],
       [[1.77229738e-07, 2.68601732e-07, 5.62901991e-07, 3.61334368e-07,
         2.13535044e-07],
        [1.83396155e-07, 2.91427877e-07, 7.09885256e-07, 4.15813189e-07,
         2.24664803e-07],
        [1.85709465e-07, 3.01607892e-07, 8.79930876e-07, 4.47673038e-07,
         2.29084998e-07],
        [1.83427421e-07, 2.91425538e-07, 7.06680623e-07, 4.15565366e-07,
         2.24695361e-07],
        [1.77224931e-07, 2.68370911e-07, 5.59722173e-07, 3.60511928e-07,
         2.13473486e-07]],
       [[1.75398267e-07, 2.65386099e-07, 5.51247442e-07, 3.56184177e-07,
         2.11202651e-07],
        [1.81699796e-07, 2.88524375e-07, 6.98556961e-07, 4.11228126e-07,
         2.22533778e-07],
        [1.84145868e-07, 2.99137885e-07, 8.77273315e-07, 4.44270026e-07,
         2.27175489e-07],
        [1.81950349e-07, 2.89169549e-07, 7.02702028e-07, 4.12528744e-07,
         2.22915144e-07],
        [1.75818229e-07, 2.66301210e-07, 5.55043114e-07, 3.57770563e-07,
         2.11801036e-07]],
       [[1.66938646e-07, 2.53055653e-07, 5.26550036e-07, 3.40034134e-07,
         2.01183284e-07],
        [1.72418890e-07, 2.74223204e-07, 6.66385134e-07, 3.91467558e-07,
         2.11296351e-07],
        [1.73990411e-07, 2.82320301e-07, 8.21770638e-07, 4.18706520e-07,
         2.14549991e-07],
        [1.71174183e-07, 2.70973774e-07, 6.47855433e-07, 3.84829848e-07,
         2.09390566e-07],
        [1.64896074e-07, 2.48648524e-07, 5.13397507e-07, 3.32670755e-07,
         1.98279413e-07]]])
    return ybar

@pytest.fixture
def my_two_component_data():
    ybar_trim = np.array([[
        [6.78795924e-07, 7.01686348e-07, 8.01859822e-07, 6.54575404e-07,
         5.86147750e-07],
        [8.93118591e-07, 1.01044963e-06, 1.08760627e-06, 9.08970737e-07,
         8.04216872e-07],
        [1.09005990e-06, 1.51395591e-06, 2.18895041e-06, 1.56156125e-06,
         1.08249656e-06],
        [1.06030173e-06, 1.35771822e-06, 1.73337573e-06, 1.51595824e-06,
         1.12148005e-06],
        [8.87440824e-07, 1.02287996e-06, 1.23583244e-06, 1.13249444e-06,
         9.56997325e-07]],
       [[6.95057611e-07, 8.35432206e-07, 1.06261004e-06, 8.99305095e-07,
         7.32804325e-07],
        [8.09742717e-07, 1.05492919e-06, 1.43766915e-06, 1.16205100e-06,
         8.66742670e-07],
        [8.75172239e-07, 1.23894490e-06, 2.14608013e-06, 1.40958716e-06,
         9.45855918e-07],
        [8.29082597e-07, 1.10288440e-06, 1.52975025e-06, 1.21627303e-06,
         8.86178288e-07],
        [7.18016977e-07, 8.75130099e-07, 1.10430275e-06, 9.41110691e-07,
         7.56015945e-07]],
       [[6.49843531e-07, 7.89512647e-07, 1.03427326e-06, 8.65798455e-07,
         6.91205926e-07],
        [7.53867148e-07, 9.89815363e-07, 1.40081086e-06, 1.10819279e-06,
         8.11863590e-07],
        [8.12491049e-07, 1.15577039e-06, 2.06803774e-06, 1.33052216e-06,
         8.82837321e-07],
        [7.69889623e-07, 1.03204022e-06, 1.48874368e-06, 1.15180122e-06,
         8.27565395e-07],
        [6.69875308e-07, 8.26185161e-07, 1.08094531e-06, 8.98515696e-07,
         7.08868996e-07]],
       [[5.38070313e-07, 6.61388554e-07, 8.92535864e-07, 7.39803311e-07,
         5.77997480e-07],
        [6.23192516e-07, 8.22422661e-07, 1.20670143e-06, 9.37011543e-07,
         6.72947869e-07],
        [6.77164279e-07, 9.65679392e-07, 1.75427227e-06, 1.11299841e-06,
         7.32227156e-07],
        [6.48558116e-07, 8.74125093e-07, 1.27402740e-06, 9.79381998e-07,
         6.95077317e-07],
        [5.66898459e-07, 7.01068275e-07, 9.27462212e-07, 7.67217788e-07,
         6.00170529e-07]]])
    kurtosis_map = np.array([
        [1.91380693, 1.55782066, 1.27171119, 1.08833282, 1.02749551,
         1.07150682, 1.11635444, 1.23115968, 1.405092  , 1.62503012],
        [2.30940445, 1.83576031, 1.42308581, 1.15675259, 1.0759083 ,
         1.13154697, 1.18255044, 1.34491972, 1.59086255, 1.88417317],
        [2.97029226, 2.39380564, 1.76191518, 1.29496091, 1.1713133 ,
         1.21681466, 1.29511235, 1.57516296, 1.94876758, 2.32885887],
        [3.82246852, 3.4337086 , 2.71641518, 1.76578055, 1.34413923,
         1.33416537, 1.6430093 , 2.19152103, 2.68168558, 3.05339968],
        [4.02059459, 3.94261714, 3.80437848, 3.50847935, 2.6304971 ,
         2.73780921, 3.23658831, 3.55078133, 3.74262018, 3.86515134],
        [3.16015004, 2.77293542, 2.23467474, 1.62505916, 1.40562588,
         1.31127397, 2.49819678, 3.38667439, 3.80341415, 3.99499944],
        [2.31046908, 1.90151357, 1.52427507, 1.30081999, 1.23387369,
         1.08291611, 1.42886235, 2.13711064, 2.82258727, 3.32086012],
        [1.83368963, 1.54303228, 1.32668847, 1.21687229, 1.11168503,
         1.03704828, 1.18182108, 1.57250558, 2.08491235, 2.59049802],
        [1.58092215, 1.37653744, 1.23203566, 1.15021943, 1.04524221,
         1.01358309, 1.09786527, 1.34131655, 1.69539686, 2.09911566]])
    return ybar_trim, kurtosis_map


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



def test_two_component_cube(
    my_component,
    my_second_component,
    my_two_component_data):
    """Checks value of ybar and kurtosis map for a two component mixture model

    """
    ssps, cube, gc1 = my_component
    gc2 = my_second_component
    my_ybar_trim, my_kurtosis_map = my_two_component_data
    cube.combine_components([gc1, gc2], [0.7, 0.3])
    ybar_trim = cube.ybar[::300, ::2, ::2]
    assert np.allclose(ybar_trim, my_ybar_trim)
    kurtosis_map = cube.get_kurtosis_v_x()
    assert np.allclose(kurtosis_map, my_kurtosis_map)




# end
