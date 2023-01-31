.. popkinmocks documentation master file, created by
   sphinx-quickstart on Tue Nov 22 14:06:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to popkinmocks's documentation!
=======================================

`popkinmocks` is a toolkit to create models of the stellar content of galaxies and produce mock observations of Integral Field Unit (IFU) datacubes.

Models are specified as distributions over stellar population and kinematic parameters, which can be created either (i) using a provided library of galactic component with parameterised forms of their morphologies, kinematics, star formation histories, and chemical enrichments, or (ii) using particle-data from simulations. Once you specify a model distribution, `popkinmocks` provides tools to evaluate the stellar integrated-light contribution to the IFU datacube, and derived properties of the distribution such as (mass or light weighted) marginal and conditional distributions, and moments.

User Guide
==================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   user/install
   user/citation.md
   user/background.md
   user/constructing_models.md
   user/derived_properties.md
   user/datacube_noise.md
   user/visualisation.md
   user/faqs.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
