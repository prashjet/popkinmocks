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

Hello!

over stellar age $t$, velocity $v$, position $\textbf{x}$ and metallicity $z$

# Statement of need

`popkinmocks` uniquely provides the ability to provide mock IFU observations of highly idealised galaxy models. The ability to create mock observations in an idealised setting is crucial for developing new methods for data analysis and inference. By stripping away complexities, the developer can understand the power and limitations of their method without confusion from undesired features in the mock data. All other existing software to produce mock IFU observations (listed below) do so using simulation particles. The inherent stochasticity in simulations is idealised-away in `popkinmocks`, which can produce mock data not only from simulation particles but also from exact, smooth models based on analytic equations.

@PNKR:2022 presented a novel inference method applied to mock data created with `popkinmocks`. Since this initial use, the software has grown into more complete library with potential for wide appeal. One unique feature of the `popkinmocks` library is its formulation in probabilistic terminology. We represent a galaxy as a probability density $p(t, v, \textbf{x}, z)$. Various moments and projections of this density are familiar objects for any galactic astrophysicist, e.g. the conditional mean of $p(v|\textbf{x})$ is more commonly called a mean velocity map. The `popkinmocks` API puts the probabilistic formulation front-and-center,

```
mean_velocity_map = galaxy.get_mean('v_x')
```

By presenting a unified view of familiar objects such as the mean velocity map $\mathbb{E}(v|\textbf{x})$, star formation history $p(t)$, or age-metallity relation $p(t,z)$, the hope is that `popkinmocks` may lower the barrier for astrophysicists to consider less familiar and higher-dimensional projections of $p(t, v, \textbf{x}, z)$.

Existing software for mock IFU observations include `simspin`[@SIMSPIN], `RealSim-IFS` [@RealSim-IFS], and the (currently private) code behind the iMaNGA project [@iMaNGA:2022]. One further novelty of `popkinmocks` compared to these tools is that we realise Doppler shifting via convolutions using Fast Fourier Transforms, while the other tools work on a particle-by-particle approach, shifting a particle's spectrum according to its velocity. This significantly different implementation of a key functionality makes `popkinmocks` a useful alternative which may be more computationally efficient in certain contexts.

# Acknowledgements

I acknowledge support from the Austrian Science Fund (FWF): F6811-N36 and from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme under grant agreement No 724857 (Consolidator Grant ArcheoDyn).

# References