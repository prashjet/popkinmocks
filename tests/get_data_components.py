import pytest
import matplotlib.pyplot as plt
import numpy as np
import popkinmocks as pkm

from conftest import my_component

def get_data_component_kinematic_maps(
    my_component,
    ):
    """Checks Delta(mass_weighted kinematic maps- light weighted kinematic maps)

    """
    ssps, cube, gc1 = my_component()
    # check get_E_v_x
    a = gc1.get_E_v_x(light_weighted=False)
    b = gc1.get_E_v_x(light_weighted=True)
    del_E_v_x = a-b
    print(repr(del_E_v_x))
    print('')
    # check get_variance_v_x
    a = gc1.get_variance_v_x(light_weighted=False)
    b = gc1.get_variance_v_x(light_weighted=True)
    del_var_v_x = a-b
    print(repr(del_var_v_x))
    print('')
    # check get_skewness_v_x
    a = gc1.get_skewness_v_x(light_weighted=False)
    b = gc1.get_skewness_v_x(light_weighted=True)
    del_skew_v_x = a-b
    print(repr(del_skew_v_x))
    print('')
    # check get_kurtosis_v_x
    a = gc1.get_kurtosis_v_x(light_weighted=False)
    b = gc1.get_kurtosis_v_x(light_weighted=True)
    del_kurt_v_x = a-b
    print(repr(del_kurt_v_x))
    print('')

def get_data_component_ybar(
    my_component,
    ):
    """Checks value of ybar for one component

    """
    ssps, cube, gc1 = my_component()
    ybar_trim = gc1.ybar[::150, ::2, ::2]
    print(repr(ybar_trim))
    print('')




get_data_component_kinematic_maps(my_component)
get_data_component_ybar(my_component)





# end
