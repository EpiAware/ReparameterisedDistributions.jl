# [Getting started](@id getting-started)

`reparameterise` wraps a distribution family so that its moments are its parameters, whichever package the family is defined in.

```@example getting-started
using ReparameterisedDistributions, Distributions
```

## A first example

A delay can be elicited as a mean and a standard deviation.
A prior belongs on that mean, not on a shape parameter that only implies it.
`reparameterise` wraps a native family so that the moments are its parameters.

```@example getting-started
d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
```

`params` reports the moments, not the native `(mu, sigma)`.
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

## Elicited quantiles

A delay is as often elicited as a pair of percentiles as it is as a mean and a
standard deviation.
The `quantiles` keyword takes `probability => value` pairs, and those values
become the parameters, so a prior goes on the quantity that was elicited.

```@example getting-started
q = reparameterise(LogNormal; quantiles = (0.05 => 1.2, 0.95 => 8.4))

(params(q), quantile(q, 0.05), quantile(q, 0.95))
```

The probabilities are arbitrary, and are carried in the wrapper's type rather
than reported as parameters.

```@example getting-started
params(reparameterise(Normal; quantiles = (1 / 3 => 2.0, 2 / 3 => 6.0)))
```

Elicitation often gives one moment and one tail point instead, so quantiles mix
with moment keywords.
The number of constraints has to match the family's native parameter count, and
an under- or over-determined request names both counts.

```@example getting-started
reparameterise(LogNormal; median = 4.0, quantiles = (0.95 => 12.0,))
```

A pair of values that falls with the probability describes no distribution at
all, so it is refused in the same way an unattainable moment is.

```@example getting-started
bad = reparameterise(Normal; quantiles = (0.25 => 3.0, 0.75 => 1.0),
    check_args = false)

logpdf(bad, 2.0)
```

## Invalid moments

Constraining each moment on its own does not always keep the combination
attainable.
A `Beta` needs `sd^2 < mean * (1 - mean)`, so a positive mean and a positive
standard deviation can still describe no `Beta` at all.
`check_args = false` turns the constructor check off, so a proposal like that
gives `logpdf == -Inf` rather than raising mid-gradient.

```@example getting-started
bad = reparameterise(Beta; mean = 0.2, sd = 0.5, check_args = false)

logpdf(bad, 0.3)
```

Every other method still converts, so an invalid distribution has no mean, no
quantile and no draw.
Asking for one raises.

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
| `Weibull` | `mean`, `sd` | numeric: the CV pins the shape by a scalar root-find; the scale then follows in closed form |
| `Weibull` | `mean`, `var` | as above, given the variance |
| `Normal`, `Logistic`, `Cauchy`, `Uniform`, `Laplace` | two of `quantiles`, `median`, `mean`, `sd` | two constraints linear in the location and scale, solved exactly |
| `LogNormal` | two of `quantiles`, `median` | as above, on the logarithms |
| `Exponential` | one of `quantiles`, `median`, `mean` | one constraint in the scale |

`Cauchy` has neither a mean nor a standard deviation, and a `LogNormal`'s are
linear in neither of its native parameters, so neither family takes those names
alongside a quantile.
Every other family falls to the generic numeric solve for constraint sets, which
is not wired up yet: a quantile elicitation of a `Gamma` is refused by name
today rather than converted.

`Weibull(mean, sd)` has no exact closed form.
Its gradient stays exact under automatic differentiation.

A family is registered with two methods, documented in [Adding a reparameterisation](@ref adding-a-reparameterisation), with the reference in [Public API](@ref public-api).

## Rescaling a moment

`rescale(d, factor)` scales one of `d`'s registered moments by `factor`,
holding the others fixed.
It routes through whichever parameterisation `d` was built under.

```@example getting-started
g = reparameterise(Gamma; mean = 8.0, shape = 2.0)

(mean(g), mean(rescale(g, 2.0)))
```

The shape stays at 2.0 and only the mean moves.
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

- Work through [Priors on moments](@ref priors-on-moments) to see what a
  prior on a moment implies in native coordinates, and how to fit one.
- Want the full interface? See the [Public API](@ref public-api).
- Registering a family of your own? See
  [Adding a reparameterisation](@ref adding-a-reparameterisation).
- Want to report a problem or ask a question? Open an issue or start a
  discussion on the [GitHub repository](https://github.com/EpiAware/ReparameterisedDistributions.jl).

The layout, navigation, and infrastructure of this site are generated by
[EpiAwarePackageTools](https://epiawarepackagetools.epiaware.org).
Its docs cover customising the generated pages and how template sync keeps
this repository current.
