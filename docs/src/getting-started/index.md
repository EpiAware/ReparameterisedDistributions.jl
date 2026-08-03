# [Getting started](@id getting-started)

`reparameterise` wraps a Distributions.jl family so that its moments are its
parameters.
This page covers what the wrapper does, which parameterisations are
registered, and how to rescale one.

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

Registering a family from another package is not supported yet: the
conversion hook is public but the validity guard is still internal, so the
contract is not one an external package can rely on. See
[#80](https://github.com/EpiAware/ReparameterisedDistributions.jl/issues/80).

The [Public API](@ref public-api) documents what is supported today.

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

## Composing with ComposedDistributions

Loading [ComposedDistributions](https://epiaware.github.io/ComposedDistributions.jl) lets a `reparameterise`d leaf sit inside a composed tree and be introspected in its registered moments, not the wrapped family's native parameters.

```julia
using ComposedDistributions

tree = compose((delay = reparameterise(LogNormal; mean = 8.0, sd = 2.0),
    tail = LogNormal(0.5, 0.4)))
params_table(tree)   # lists `mean`/`sd` for `delay`, not `mu`/`sigma`

est = uncertain(tree; delay = (mean = LogNormal(log(8.0), 0.2),))
build_priors(params_table(est)).delay.mean   # the prior sits on the mean
```

A prior or a value on a native parameter of the wrapped family (`sigma`, here) is rejected: it is not one of this leaf's parameters.

```julia
uncertain(tree; delay = (sigma = LogNormal(0.0, 1.0),))   # ArgumentError
```

The flat codec (`flatten`/`unflatten`/`reconstruct`) does not yet round-trip in moment coordinates for a `Reparameterised` leaf; see [NEWS.md](https://github.com/EpiAware/ReparameterisedDistributions.jl/blob/main/NEWS.md) for the current status and the tracking issue.

## Learning more

- Work through [Priors on moments](@ref priors-on-moments) to see what a
  prior on a moment implies in native coordinates, and how to fit one.
- Want the full interface? See the [Public API](@ref public-api).
- Want to report a problem or ask a question? Open an issue or start a
  discussion on the [GitHub repository](https://github.com/EpiAware/ReparameterisedDistributions.jl).

The layout, navigation, and infrastructure of this site are generated by
[EpiAwarePackageTools](https://epiawarepackagetools.epiaware.org).
Its docs cover customising the generated pages and how template sync keeps
this repository current.
