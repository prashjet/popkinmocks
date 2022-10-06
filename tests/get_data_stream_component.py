import pytest
import matplotlib.pyplot as plt
import numpy as np
import popkinmocks as pkm

from conftest import my_stream_component

def test_component_ybar(
    my_stream_component
    ):
    """Checks value of ybar for one component

    """
    cube, stream = my_stream_component()
    ybar_trim = stream.ybar[::150, ::2, ::2]
    print(repr(ybar_trim))
    print('')

test_component_ybar(my_stream_component)



# end
