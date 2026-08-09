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
| `Weibull` | `mean`, `sd` | numeric: the CV pins the shape by a scalar root-find; the scale then follows in closed form |
| `Weibull` | `mean`, `var` | as above, given the variance |

`Weibull(mean, sd)` has no exact closed form.
The coefficient of variation depends on the shape alone, through a strictly
monotone one-dimensional equation that a bracketing root-find solves, with
the scale then following in closed form.
The gradient is still exact: the root's derivative is recovered afterwards by
an implicit-function-theorem correction rather than by differentiating through
the solver.

Registering a family, analytic or numeric, needs two methods: a
`valid_moments` predicate stating which parameter values describe a member
of the family, and a `to_native` method that assumes that predicate has
already passed and performs the conversion unconditionally. A family with
no exact closed form calls `solve_moment` from inside its own `to_native`,
passing its moment equation, its derivative and a bracket as ordinary
functions; `valid_moments` still states the window the root-find can
actually solve, checked before the solve runs.

Keeping the two separate, rather than folding the guard into `to_native`
and returning `nothing` for an invalid point, keeps `to_native` returning a
concretely-typed distribution rather than a `Union` of one and `nothing` —
which matters on the hot path a gradient runs through, not only for style.

Register both methods under the parameter names sorted alphabetically:
`reparameterise` canonicalises its keywords that way before dispatching, so
a method registered under `Val((:sd, :mean))` is never found. The
[Public API](@ref public-api) documents both hooks, and
[Adding a reparameterisation](@ref adding-a-reparameterisation) is the full
contract, with a worked example and the test suite a registration is
checked against.

### Migrating a family registered against v0.2.0

`v0.2.0` folded the guard into `to_native` itself, which returned
`Union{Nothing, D}` instead of a concrete `D` for an invalid point. That
shape let a differentiated call site bind a `Union{Nothing, D}`-typed
value, which produced a silently wrong reverse-mode gradient under Enzyme
for some parameterisations.

A family registered against `v0.2.0` has a `to_native` method that guards
its own input and returns `nothing` for an invalid point. Move that guard
into a `valid_moments` method instead, and let `to_native` run
unconditionally on parameters already known valid. Left unmoved, the
registration fails loudly rather than silently: the 3-arg `valid_moments`
fallback reports the point valid, `to_native` returns `nothing`, and the
density raises a `MethodError` on `nothing` at exactly the point that used
to give `-Inf`.

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
