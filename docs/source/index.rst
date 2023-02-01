.. popkinmocks documentation master file, created by
   sphinx-quickstart on Tue Nov 22 14:06:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to popkinmocks's documentation!
=======================================

`popkinmocks` is a toolkit to create models of the stellar content of galaxies and produce mock observations of Integral Field Unit (IFU) datacubes.

Models in `popkinmocks` are distributions over stellar population and kinematic parameters. These can be created either using particle data from simulations, or a provided library of parametric galactic components. Given a model, `popkinmocks` provides tools to evaluate the stellar integrated-light contribution to the IFU datacube, and derived properties of the distribution such as (mass or light weighted) moments, and conditional and marginal probability functions.

User Guide
==================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   user/install
   user/citation.md
   user/background.md
   user/constructing_models.md
   user/moments_prob_func.md
   user/datacube_noise.md
   user/visualisation.md
   user/faqs.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
