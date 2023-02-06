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

# Probability Functions \& Moments 

Once you've constructed a model of $p(t, v, \textbf{x}, z)$, _popkinmocks_ provides tools to calculate its marginal and conditional probability functions, and moments. These can be either light weighted or mass weighted.

Let's take a look, using the mixture model saved in the [Constructing Models](constructing_models.md) page:

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
import dill
import popkinmocks as pkm
with open('data/my_mixture_component.dill', 'rb') as file:
    galaxy = dill.load(file)
cube = galaxy.cube
```

## Probability Functions

All probability functions can be evaluated using a component's `get_p` method. For example, to evaluate the marginal distribution

$$
p(z) = \int\int\int 
  p(t, v, \textbf{x}, z)
  \;\mathrm{d}t \;\mathrm{d}v \;\mathrm{d}\textbf{x}.
$$

do this,

```{code-cell}
p_z = galaxy.get_p('z')
```

while to evaluate the conditional distribution

$$
p(v|t) = \frac{p(v,t)}{p(t)}
$$

do this

```{code-cell}
p_v_t = galaxy.get_p('v_t')
p_v_t.shape
```

The shape of the returned array corresponds to the order of the variables in the input string. Rules for constructing valid strings accepted by `get_p` are [below](dist_strings).

`get_p` accepts two optional keyword arguments:

- `density` determines whether a probability density is returned (`True`, by default) or an integrated probability (`False`)
- `light_weighted` (default `False`), see [below](light_weight) for details.

So to evaluate light-weighted LOSVDs conditional on position, do this

```{code-cell}
p_v_x = galaxy.get_p('v_x', light_weighted=True, density=False)
p_v_x.shape
```

Since `density=False`, these should be close to one when summed over the velocity dimension,

```{code-cell}
np.allclose(np.sum(p_v_x, 0), 1.)
```

`Mixture` components can additionally take a third argument `collapse_cmps` (default `True`). If `collapse_cmps=False`, then the returned distribution is split by mixture components, e.g.

```{code-cell}
p_x = galaxy.get_p('x', light_weighted=True, density=False, collapse_cmps=False)
p_x.shape
```

The first dimension has size two corresponding to the two components in this mixture model. They are weighted by the mixture weights, so the total probability still sums to one,

```{code-cell}
np.sum(p_x)
```

Lastly note there is an equivalent (and more numerically accurate) version `get_log_p` for evaluating log probabilties.

(dist_strings)=
## Distribution Strings \& Shapes

Distributions are specified by a string. The rules for constructing these strings are:

- marginal distributions have no underscore
- conditional distributions have an underscore dividing dependent variables on the left and conditioners on the right
- variables must be in alphabetical order (on either side of the undescore, if present)

Some examples:

- `tz` $\rightarrow p(t, z)$
- `vt` invalid since not in alphabetical order 
- `xz_v` $\rightarrow (\textbf{x}, z | v)$
- `zx_v` invalid since `zx` is not in alphabetical order 

The order of dimensions in the arrays returned by `get_p` correspond to the order of the variables provided in the input string.

## Moments

To evaluate moments e.g.

$$
\mathbb{E}(z) = \int z p(z) \;\mathrm{d}z,
$$

do this 

```{code-cell}
mean_z = galaxy.get_mean('z')
print(mean_z)
```

To calculate conditional moments e.g.

$$
\mathbb{Var}(v|t) = \int (v - \mathbb{E}(v|t))^2 p(v|t) \;\mathrm{d}v.
$$

do this 

```{code-cell}
var_v_t = galaxy.get_variance('v_t')
cube.plot('t', var_v_t**0.5)
_ = plt.gca().set_ylabel('Velocity dispersion [km/s]')
```

There are four main methods for evaluating moments:

- `get_mean`,
- `get_variance`,
- `get_skewness`,
- `get_kurtosis`,

which are all wrappers around the underlying methods `get_central_moment` and `get_noncentral_moment`.

Rules for constructing valid input strings are similar to those given [above](dist_strings). One difference is that moments are only defined for distributions over one variable i.e. it makes sense to talk about the variance of $p(z)$ but not of $p(t,z)$ since we don't know whether the variance is with respect to $t$ or $z$. For this reason `get_variance('z')` is valid while `get_variance('tz')` is not. Higher order cross-moments (e.g. covariance) will be implemented in the future.

In general, the returned moment is scalar if the input distribution is marginal, or has the same shape as the conditioners if the input distribution is conditional. There is an exception for the vector variable $\textbf{x}=(x_1,x_2)$, which will return two values corresponding to $(x_1, x_2)$,

```{code-cell}
mean_x = galaxy.get_mean('x')
print(mean_x)
```

or an array whose first dimension has size of two for conditional moments,

```{code-cell}
var_x_t = galaxy.get_variance('x_t')
print(var_x_t.shape)

# var_x_t[0] = Var(x_1|t)
# var_x_t[1] = Var(x_2|t)
```

As before, you can also evaluate light weighted quantities e.g. to evaluate a skewness map of the light-weighted LOSVDs,

```{code-cell}
skew_v_x = galaxy.get_skewness('v_x', light_weighted=True)
_ = cube.imshow(skew_v_x)
```

(light_weight)=
## Light or Mass Weighting

By default _popkinmocks_ deals with mass-weighted quantities. When we say the density $p(t, v, \textbf{x}, z)$ is mass-weighted, this means that if you integrate it over some volume in $(t, v, \textbf{x}, z)$ space, the result is equal to the fraction of stellar *mass* in that volume. The light-weighted distribution, on the other hand, would return the fraction of stellar *light*.

Mass-weighted and light-weighted quantities are different because different stellar populations have different integrated luminosities,

$$
\mathcal{L}(t,z) = \int S(\lambda ; t, z) \;\mathrm{d}t \;\mathrm{d}z
$$

```{code-cell}
ssps = pkm.model_grids.milesSSPs()
ssps.get_light_weights()

from matplotlib.colors import LogNorm

ax = cube.imshow(
  ssps.light_weights,
  view=['t','z'],
  norm=LogNorm(),
  colorbar_label='$\mathcal{L}(t,z)$'
  )
_ = ax.set_title('Integrated Flux of SSPs')
```

This varies strongly with age - young stars are much brighter than old stars - and weakly with metallicty.

We get the light-weighted density $p_\mathcal{L}(t, v, \textbf{x}, z)$ from the mass-weighted version by scaling by light weights and appropriately normalising, i.e.

$$
p_\mathcal{L}(t, v, \textbf{x}, z) \propto \mathcal{L}(t,z) p(t, v, \textbf{x}, z).
$$

This $p_\mathcal{L}(t, v, \textbf{x}, z)$ forms the basis for calculating all light-weighted distributions and moments.