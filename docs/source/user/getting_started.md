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

# Getting started

Some basic background 

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

# get t, z and spectrum for a given ssp index
ssp_id = 40
id_z, id_t = ssps.par_idx[:,ssp_id]
z = ssps.par_cents[0][id_z]
t = ssps.par_cents[1][id_t]
spectrum = ssps.X[:,ssp_id]

# plot it
_ = plt.plot(ssps.lmd, spectrum)
_ = plt.gca().set_title(f'SSP with age {t} Gyr & metallicity {z} [M/H]')
_ = plt.gca().set_xlabel('Wavelength [Ang.]')
_ = plt.gca().set_ylabel('Flux')
```

Stellar kinematics refer to the velocities and positions of stars within a galaxy. By describing stellar populations and kinematics simultaneously via the joint distribution $p(t, v, \textbf{x}, z)$, we can capture all relations between these four variables without imposing simplifying assumptions. One such assumption which is commonly used when modelling observed spectra is that at a fixed position, velocities and stellar populations are independent. This statement is equivalent to the equation:

$$
p(t, v, \textbf{x}, z) = p(x) p(v|\textbf{x}) p(t,z|\textbf{x})
$$

which factorises the joint distribution into marginal and conditional terms, where velocities and populations only interact via their dependence on position.

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

The two factors of $(1+v/c)$ ain this equation rise from Doppler-shifting of light: the factor inside $S$ translates from rest- to observed-wavelengths given a LOS velocity $v$, while the pre-factor of scales the flux to ensure total luminosity is conserved. The integrals over $t$ and $z$ allow the 

## Evaluating datacubes using FFTs

_popkinmocks_ evaluates equation $\eqref{eq:fwdmod}$ to produce datacubes for a given choice of $p(t, v, \textbf{x}, z)$. While the integrals over age and metallicity are straightforward, the $v$ integral is more complicated. Rather than evaluate the $v$ integral directly, _popkinmocks_ uses the fact that \eqref{eq:fwdmod} can be transformed into a standard convolution by changing variables:

* log wavelength, $\omega = \ln \lambda$
* transformed velocity, $u = \ln(1 + v/c)$.

This transforms $\eqref{eq:fwdmod}$ into a standard convolution over $u$:

$$
\begin{equation}
  y(\textbf{x}, \omega) = \int \int \int 
    \tilde{p}(t, u, \textbf{x}, z) \tilde{S}\left(\omega - u; t, z\right) 
    \; \mathrm{d}t \; \mathrm{d}u \; \mathrm{d}z
  \label{eq:fwdmod_transformed}
\end{equation}
$$

where $\tilde{p}$ and $\tilde{S}$ are re-labellings of $p$ and $S$ (see Section 2.2.1 of [Ocvirk et. al](https://ui.adsabs.harvard.edu/abs/2006MNRAS.365...74O/abstract) for details). _popkinmocks_ evaluates this convolution using a Fast Fourier Transform (FFT) algorithm, producing a datacube sampled in log wavelength $\omega$ rather than wavelength $\lambda$.

## Discretisation


```{code-cell}
cube = pkm.ifu_cube.IFUCube(
  ssps=ssps, 
  nx=25, 
  ny=25, 
  nv=50, 
  vrng=(-1000,1000))
cube.get_distribution_shape('tvxz')
```