import pytest
import matplotlib.pyplot as plt
import numpy as np
import popkinmocks as pkm

from conftest import my_three_component_cube
from conftest import my_component, my_second_component, my_stream_component

def get_data_three_component_cube_ybar(cube):
    ybar_trim = cube.ybar[::100,::3,::2]
    print(repr(ybar_trim))
    print('')

get_data_three_component_cube_ybar(
    my_three_component_cube(
        my_component,
        my_second_component,
        my_stream_component)
    )

# end
