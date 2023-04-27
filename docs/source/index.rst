.. popkinmocks documentation master file, created by
   sphinx-quickstart on Tue Nov 22 14:06:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

popkinmocks
=======================================

`popkinmocks` is a toolkit to create models of the stellar content of galaxies and produce mock observations of Integral Field Unit (IFU) datacubes.

Models in `popkinmocks` are distributions over stellar population and kinematic parameters. These can be created either using particle data from simulations, or a provided library of parametric galactic components. Given a model, `popkinmocks` provides tools to evaluate the stellar integrated-light contribution to the IFU datacube, and derived properties of the distribution such as (mass or light weighted) moments, and conditional and marginal probability functions.

`popkinmocks` is being actively developed in `a public repository on GitHub
<https://github.com/prashjet/popkinmocks>`_. If you have any trouble, `open
an issue <https://github.com/prashjet/popkinmocks/issues>`_.
Information about contributing can be found `here
<user/contributing.html>`_.

Contents
---------------------

.. toctree::
   :maxdepth: 1
   :caption: User Guide:

   user/install
   user/citation.md
   user/contributing.md
   user/code_of_conduct.md

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   user/background.md
   user/constructing_models.md
   user/moments_prob_func.md
   user/datacube_noise.md
   user/visualisation.md
   user/faqs.md

.. toctree::
   :maxdepth: 1
   :caption: API Documentation:

   api/cube_and_ssps
   api/components
   api/noise

License & attribution
---------------------

Copyright 2023 Prashin Jethwa.

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite this package and its dependencies. You
can find more information about how and what to cite in the `citation
<user/citation.html>`_ documentation.
