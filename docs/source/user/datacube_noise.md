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

# Observation Noise

Here we present noise models available in _popkinmocks_. Noise $\epsilon$ is added to the signal $\bar{y}$ to give the observed cube $y_\mathrm{obs}$, i.e.

$$
y_\mathrm{obs}(\mathbf{x},\omega) = 
  \bar{y}(\mathbf{x},\omega) + \epsilon(\mathbf{x},\omega).
$$

Currently we assume that noise is un-correlated between spaxels and Gaussian i.e. $\epsilon(\mathbf{x},\omega)$ is sampled from a normal distribution with variance $\sigma(\mathbf{x},\omega)^2$, i.e.

$$
\epsilon(\mathbf{x},\omega) \sim \mathcal{N}(0, \sigma(\mathbf{x},\omega)^2)
$$

We provide two models for $\sigma(\mathbf{x},\omega)$. We demonstrate these on the mixture model saved in the [Constructing Models](constructing_models.md) page:

```{code-cell}
import numpy as np
np.random.seed(301288)
import matplotlib.pyplot as plt
import dill

import popkinmocks as pkm

with open('data/my_mixture_component.dill', 'rb') as file:
    galaxy = dill.load(file)
cube = galaxy.cube
```

## Constant SNR

This implements a constant signal-to-noise ratio (SNR), i.e.

$$
\sigma(\mathbf{x},\omega) = \frac{\bar{y}(\mathbf{x},\omega)}{\mathrm{SNR}}
$$

```{code-cell}
constant_snr = pkm.noise.ConstantSNR(galaxy)
yobs_const_snr = constant_snr.get_noisy_data(snr=100)
```

## Shot noise

This model is more realistic. Motivated by Poisson/shot noise, here the noise standard deviation scales as the square root of the signal, i.e.

$$
\sigma(\mathbf{x},\omega) = K \sqrt{\bar{y}(\mathbf{x},\omega)}.
$$

The constant of proportionality is chosen so that a desired maximum signal-to-noise ratio is achieved in the brightest voxel, i.e.

$$
K = \frac{\max{\sqrt{\bar{y}(\mathbf{x},\omega)}}}{\mathrm{SNR}}.
$$

```{code-cell}
shot_noise = pkm.noise.ShotNoise(galaxy)
yobs_shot_noise = shot_noise.get_noisy_data(snr=100)
```

## Comparison

```{code-cell}
def plot_spectrum_from_pixel(i, j, ax):
  cube.plot_spectrum(galaxy.ybar[:,i,j], '-', ax=ax, label='$\\bar{y}$')
  cube.plot_spectrum(yobs_const_snr[:,i,j], '.', ax=ax, label='ConstantSNR')
  cube.plot_spectrum(yobs_shot_noise[:,i,j], '.', ax=ax, label='ShotNoise')
  ax.legend()
  ax.set_xlim(5800, 6000)
  ax.set_yticks([])

fig, ax = plt.subplots(1,2)
plot_spectrum_from_pixel(10, 10, ax=ax[0])
ax[0].set_title('Inner, bright pixel')
plot_spectrum_from_pixel(0, 0, ax=ax[1])
ax[1].set_title('Outer, faint pixel')
fig.tight_layout()
```

In the bright central pixel (left) the SNRs of the two noise models are equal. At the outer pixel (right) the SNR of the `ShotNoise` model has decreased in line with the decreased flux, while for `ConstantSNR` it remains high.