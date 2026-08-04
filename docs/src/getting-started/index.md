# [Getting started](@id getting-started)

Welcome to the `ReparameterisedDistributions` documentation.
This page is the quickstart.
The home page is generated from the README and already carries the install
instructions and a worked fit, so this page picks up from there: what the
wrapper does, which parameterisations exist, and how to move between them.

```@example getting-started
using ReparameterisedDistributions, Distributions
```

## A first example

A delay is elicited as a mean and a standard deviation.
A prior belongs on that mean, not on a shape parameter that only implies it.
`reparameterise` wraps a native family so that the moments are its parameters.

```@example getting-started
d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
```

`params` reports the moments, not the native `(mu, sigma)` that only implies
them.
Every other method works exactly as it would on the native distribution.

```@example getting-started
params(d), mean(d), std(d), logpdf(d, 7.5)
```

The wrapper takes its variate form and value support from the family it wraps,
so a wrapped discrete family stays discrete.

```@example getting-started
nb = reparameterise(NegativeBinomial; mean = 10.0, overdispersion = 0.5)

(mean(nb), var(nb))
```

## Invalid moments

A sampler exploring an unconstrained space will propose moments no member of
the family can have.
`check_args = false` turns the constructor check off, so an invalid proposal
gives `logpdf == -Inf` rather than raising mid-gradient.

```@example getting-started
bad = reparameterise(Gamma; mean = 1.0, sd = -1.0, check_args = false)

logpdf(bad, 2.0)
```

Every other method still converts, so an invalid distribution has no mean, no
quantile and no draw.
Asking for one raises rather than returning a number that means nothing.

## Supported parameterisations

| Family | Parameters | Conversion |
|---|---|---|
| `LogNormal` | `mean`, `sd` | the moments of the distribution, not of its logarithm |
| `LogNormal` | `mean`, `var` | as above, given the variance |
| `Gamma` | `mean`, `sd` | `scale = var / mean`, `shape = mean² / var` |
| `Gamma` | `mean`, `var` | as above, given the variance |
| `Gamma` | `mean`, `shape` | `scale = mean / shape`; the shape is native |
| `Gamma` | `shape`, `rate` | `scale = 1 / rate`; the shape is native |
| `NegativeBinomial` | `mean`, `overdispersion` | `var = mean + overdispersion · mean²` |
| `NegativeBinomial` | `mean`, `dispersion` | `var = mean + mean² / dispersion`, the reciprocal convention |
| `Exponential` | `rate` | `scale = 1 / rate` |
| `SkewNormal` | `centre`, `scale`, `mass_below_centre` | `alpha = tan(π · (1/2 − mass_below_centre))` |
| `Beta` | `mean`, `sd` | `nu = mean·(1−mean)/var − 1`; `alpha = mean·nu`, `beta = (1−mean)·nu` |
| `Beta` | `mean`, `var` | as above, given the variance |
| `InverseGaussian` | `mean`, `sd` | `lambda = mean³ / var`; the mean is native |
| `InverseGaussian` | `mean`, `var` | as above, given the variance |

The `NegativeBinomial` parameterisations are the two epidemiology reaches for.
The overdispersion is the excess variance relative to a Poisson, so it tends to
the Poisson limit as it goes to zero.
The dispersion is its reciprocal, so it tends to the Poisson limit as it goes
to infinity instead.

The `Exponential` and `Gamma` rate parameterisations let a distribution be
specified, reported and estimated directly by its rate rather than by a scale
hand-inverted from it, the natural coordinates for a hazard.

The `SkewNormal` parameterisation is keyed on an elicitation quantity rather
than a moment, the probability mass falling below a reference point.
That mass depends only on the native shape for the untruncated family, so it
inverts exactly, and the elicited fraction is exact only before any truncation.
Distributions.jl does not implement `cdf` or `quantile` for `SkewNormal`, a
limitation this parameterisation inherits rather than works around.

`Beta(mean, sd)` is the natural coordinates for a probability-scale quantity
elicited as a central value and an uncertainty, a reporting fraction or a
case-fatality ratio say.
A Beta's variance cannot exceed `mean * (1 - mean)`, the variance of a
Bernoulli with the same mean.
A standard deviation too wide for its mean has no Beta at all, and the validity
guard rejects it rather than silently clipping it to the boundary.

Adding a family is one `to_native` method for the closed form and one
`_valid_moments` method for the guard, so a downstream package can register its
own.
See the [Public API](@ref public-api) and the Internal API page in the sidebar.

## Rescaling a moment

`rescale(d, factor)` scales one of `d`'s registered moments by `factor`,
holding the others fixed.
It routes through whichever parameterisation `d` was built under.

```@example getting-started
g = reparameterise(Gamma; mean = 8.0, shape = 2.0)

(mean(g), mean(rescale(g, 2.0)))
```

The shape stays where it was; only the named moment moves.
A discrete family scales in moment coordinates rather than by an affine
transform of the native support.

```@example getting-started
mean(rescale(nb, 3.0))
```

`parameter` defaults to `:mean` and can name any of `d`'s registered
parameters.
Naming one that is not registered for `d`'s family raises a `DomainError`
rather than applying the factor under different semantics.

## Learning more

- Work through [Priors on moments](@ref priors-on-moments) to see why a prior
  on a mean cannot be written with native parameters, and how to fit one.
- Want the full interface? See the [Public API](@ref public-api).
- Want to report a problem or ask a question? Open an issue or start a
  discussion on the [GitHub repository](https://github.com/EpiAware/ReparameterisedDistributions.jl).

The layout, navigation, and infrastructure of this site are generated by
[EpiAwarePackageTools](https://epiawarepackagetools.epiaware.org).
Its docs cover customising the generated pages and how template sync keeps
this repository current.
