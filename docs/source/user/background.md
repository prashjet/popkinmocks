---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# IFUs for Galaxy Stellar Light

This introduction provides some background about stellar populations, kinematics and the forward model to evaluate the integrated-light stellar contribution to IFU datacubes.

## Stellar populations and kinematics

_popkinmocks_ describes the stellar content of galaxies as a joint probability distribution $p(t, v, \textbf{x}, z)$ over four variables:

* age $t$,
* line-of-sight (LOS) velocity $v$,
* 2D on-sky position $\mathbf{x}$, and
* metallicity $z$.

A simple (AKA single) stellar population (SSP) is a population of stars born at the same time and having the same initial chemical composition. In _popkinmocks_ we use a specific set of models ([Miles](http://miles.iac.es/pages/ssp-models.php)) for the spectra of SSPs. Let's look at the spectrum of one SSP:

```{code-cell}
import popkinmocks as pkm
import matplotlib.pyplot as plt

ssps = pkm.model_grids.milesSSPs()

# plot ssp for a given model index
index = 40
t, z, spectrum = ssps.get_ssp_wavelength_spacing(index)
_ = plt.plot(ssps.lmd, spectrum)
_ = plt.gca().set_title(f'SSP with age {t} Gyr & metallicity {z} [M/H]')
_ = plt.gca().set_xlabel('Wavelength [Ang.]')
_ = plt.gca().set_ylabel('Flux')
```

Stellar kinematics refer to the velocities and positions of stars within a galaxy. By describing stellar populations and kinematics simultaneously via the joint distribution $p(t, v, \textbf{x}, z)$, we can capture all relations between these four variables.

## IFU datacubes

Integral Field Units (IFUs) are a type of instrument which can observe spatial and spectral information simultaneously. IFU observations result in datacubes $y(\textbf{x}, \lambda)$ which describe the flux observed at a position $\textbf{x}$ and wavelength $\lambda$. The datacube is connected to the stellar population-kinematic distribution via the equation:

$$
\begin{equation}
  y(\textbf{x}, \lambda) = \int \int \int 
    \frac{1}{1+v/c} p(t, v, \textbf{x}, z) S\left(\frac{\lambda}{1+v/c} ; t, z\right) 
    \; \mathrm{d}t \; \mathrm{d}v \; \mathrm{d}z
  \tag{1}
  \label{eq:fwdmod}
\end{equation}
$$

where we have introduced notation for the:

* spectrum of SSP with age $t$ and metallicity $z$, $S(\lambda ; t, z)$, and
* speed of light, $c$.

The two factors of $(1+v/c)$ in this equation arise from Doppler-shifting of light: the factor inside $S$ translates from the rest-frame to observed wavelengths given a LOS velocity $v$, while the pre-factor scales the flux, ensuring total luminosity is conserved. To see how equation $\eqref{eq:fwdmod}$ connects to the standard equation for spectral modelling [see here](faq_spec_modelling).

## Evaluating datacubes using FFTs

_popkinmocks_ evaluates equation $\eqref{eq:fwdmod}$ to produce datacubes for a given choice of $p(t, v, \textbf{x}, z)$. To help perform the integral over $v$, _popkinmocks_ uses the fact that $\eqref{eq:fwdmod}$ can be re-written as a standard convolution by changing variables:

* log wavelength, $\omega = \ln \lambda$
* transformed velocity, $u = \ln(1 + v/c)$.

This transforms $\eqref{eq:fwdmod}$ into a standard convolution over $u$:

$$
\begin{equation}
  \tilde{y}(\textbf{x}, \omega) = \int \int \int 
    \tilde{p}(t, u, \textbf{x}, z) \tilde{S}\left(\omega - u; t, z\right) 
    \; \mathrm{d}t \; \mathrm{d}u \; \mathrm{d}z
  \label{eq:fwdmod_transformed}
\end{equation}
$$

where $\tilde{y}$, $\tilde{p}$ and $\tilde{S}$ are trivial re-labellings (see Section 2.2.1 of [Ocvirk et. al](https://ui.adsabs.harvard.edu/abs/2006MNRAS.365...74O/abstract) for details). _popkinmocks_ evaluates this convolution using a Fast Fourier Transform (FFT), producing a datacube sampled in log wavelength $\omega$ rather than wavelength $\lambda$.

## Discretisation

The discretisation in age and metallicity is set by the SSP models; there are various options when instantiating SSP models to modify the range and sampling of these parameters. Discretisation in the spatial dimensions $\textbf{x}=(x_1, x_2)$ and velocity $v$ are chosen when instantiating the `IFUCube` object,

```{code-cell}
cube = pkm.ifu_cube.IFUCube(
  ssps=ssps, 
  nx1=20, x1rng=(-1,1),      # arbitrary units
  nx2=21, x2rng=(-1,1),      # arbitrary units
  nv=30, vrng=(-1000,1000)   # km/s
)
cube.get_distribution_shape('tvxz')
```

This shape corresponds to the size of each variable in the string `tvxz` i.e. $(t, v, x_1, x_2, z)$. To see the values of metallicity, for example:

```{code-cell}
cube.get_variable_values('z')
```