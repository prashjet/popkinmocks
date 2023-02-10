---
title: 'popkinmocks: mock IFU datacubes for modelling stellar populations and kinematics.'
tags:
  - Python
  - astronomy
  - spectroscopy
  - stellar-populations
  - stellar-kinematics
authors:
  - name: Prashin Jethwa
    orcid: 0000-0003-0010-8129
    affiliation: "1" 
affiliations:
 - name: Department of Astrophysics, University of Vienna, Türkenschanzstraße 17, A-1180 Vienna, Austria
   index: 1
date: 8 February 2023
bibliography: paper.bib

---

# Summary

Integral Field Units (IFUs) are a type of detector which measure both spatial and spectral information. These detectors output 3D data-products known as datacubes: 2D images with a spectrum associated with each image pixel. Beyond the very local Universe where individual stars can be resolved, IFU datacubes are our most information-rich datasets of galaxies, and large samples of these have been observed [e.g. @SAMI_Croom2021].

Galaxies are built from stars, gas, dust and dark matter. Stars are an especially useful observational probe as many physical properties can be inferred from their spectra. _Stellar populations_ are intrinsic properties such as a star's age or chemical composition, which affect the strengths of spectral absorption lines. _Stellar kinematics_ are the positions and velocities of stars within the galaxy. The component of velocity along the line-of-sight produces Doppler shifting of absorption lines. In the past, stellar populations and kinematics of external galaxies have been modelled separately; often thought of as distinct subfields. Recent work has demonstrated the power of moving beyond this dichotomy to study galaxy evolution [e.g. @Poci2019]. `popkinmocks` is software to create mock observations of IFU datacubes of galaxy stellar light for the era of combined population-kinematic analyses.

# Statement of need

Combined population-kinematic models require a framework that can self-consistently describe interactions between populations and kinematics. To this end, `popkinmocks` has been formulated in probabilistic language where the galaxy is represented as a joint probability density $p(t, v, \textbf{x}, z)$ over stellar age $t$, velocity $v$, position $\textbf{x}$ and metallicity $z$. This joint distribution can encode all interaction between these variables. Projections of this distribution are familiar quantities for any galactic astrophysicist, e.g. the 1D marginal $p(t)$ is the _star formation history_, the 2D marginal $p(t,z)$ is the _age-metallity relation_, the conditional mean of $p(v|\textbf{x})$ is the _mean velocity map_. The `popkinmocks` API puts this unified, probabilistic formulation front-and-center, e.g.

```
mean_velocity_map = galaxy.get_mean('v_x')
```

This framework elucidates underlying connections between various quantities e.g. it is used in the [documentation](https://popkinmocks.readthedocs.io/en/stable/user/background.html#how-is-this-connected-to-spectral-modelling) to explain a common assumption used in spectral modelling. Furthermore, this unified formulation may promote exploration of higher-order cross-moments which combine population and kinematics in novel ways.

Another unique feature of `popkinmocks` is the ability to create mock IFU observations of highly idealised galaxy models. Idealised models are useful as they provide a controlled setting to develop new inference methods. All existing software to produce mock IFU observations (listed below) use stellar particles from simulations, which are inherently stochastic. In addition to simulation particles, `popkinmocks` can also create mock observations using smooth galaxy models based on analytic equations. These smooth models were used in @PNKR:2022 to test a novel inference method, and ongoing development of this method will continue to make significant use of `popkinmocks`.

Existing software for mock IFU observations include `simspin` [@SIMSPIN], `RealSim-IFS` [@RealSim-IFS], and the (currently private) code behind the iMaNGA project [@iMaNGA:2022]. A key functionality needed to create mock IFU datacubes is Doppler shifting of spectra. All three existing tools implement this functionality in the same way: using a particle-by-particle approach, shifting a particle's spectrum according to its velocity. By contrast, in `popkinmocks` we realise Doppler shifting via convolutions using Fast Fourier Transforms. This significantly different implementation of a key functionality makes `popkinmocks` a useful alternative which may be more computationally efficient in certain contexts.

# Acknowledgements

I acknowledge support from the Austrian Science Fund (FWF): F6811-N36 and from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme under grant agreement No 724857 (Consolidator Grant ArcheoDyn). I also thank Stefanie Reiter and Christine Ackerl for useful advice.

# References