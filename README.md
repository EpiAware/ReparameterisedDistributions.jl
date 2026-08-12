# ReparameterisedDistributions.jl <img src="docs/src/assets/logo.svg" width="150" alt="ReparameterisedDistributions logo" align="right">

<!-- badges:start -->
| **Documentation** | **Build Status** | **Code Quality** | **License & DOI** | **Downloads** |
|:-----------------:|:----------------:|:----------------:|:-----------------:|:-------------:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://reparameteriseddistributions.epiaware.org/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://reparameteriseddistributions.epiaware.org/dev/) | [![Test](https://github.com/EpiAware/ReparameterisedDistributions.jl/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/EpiAware/ReparameterisedDistributions.jl/actions/workflows/test.yaml) [![codecov](https://codecov.io/gh/EpiAware/ReparameterisedDistributions.jl/graph/badge.svg)](https://codecov.io/gh/EpiAware/ReparameterisedDistributions.jl) [![AD](https://github.com/EpiAware/ReparameterisedDistributions.jl/actions/workflows/ad.yaml/badge.svg?branch=main)](https://github.com/EpiAware/ReparameterisedDistributions.jl/actions/workflows/ad.yaml) | [![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![JET](https://img.shields.io/badge/%E2%9C%88%EF%B8%8F%20tested%20with%20-%20JET.jl%20-%20red)](https://github.com/aviatesk/JET.jl) | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) | [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FReparameterisedDistributions&query=total_requests&label=Downloads)](https://juliapkgstats.com/pkg/ReparameterisedDistributions) [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FReparameterisedDistributions&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/ReparameterisedDistributions) |

| ForwardDiff | ReverseDiff (tape) | ReverseDiff (compiled) | Enzyme forward | Enzyme reverse | Mooncake reverse | Mooncake forward |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [![cov ForwardDiff](https://codecov.io/gh/EpiAware/ReparameterisedDistributions.jl/graph/badge.svg?flag=ad-forwarddiff)](https://app.codecov.io/gh/EpiAware/ReparameterisedDistributions.jl?flags%5B0%5D=ad-forwarddiff) | [![cov ReverseDiff](https://codecov.io/gh/EpiAware/ReparameterisedDistributions.jl/graph/badge.svg?flag=ad-reversediff)](https://app.codecov.io/gh/EpiAware/ReparameterisedDistributions.jl?flags%5B0%5D=ad-reversediff) | [![cov ReverseDiff compiled](https://codecov.io/gh/EpiAware/ReparameterisedDistributions.jl/graph/badge.svg?flag=ad-reversediff-compiled)](https://app.codecov.io/gh/EpiAware/ReparameterisedDistributions.jl?flags%5B0%5D=ad-reversediff-compiled) | [![cov Enzyme forward](https://codecov.io/gh/EpiAware/ReparameterisedDistributions.jl/graph/badge.svg?flag=ad-enzyme-forward)](https://app.codecov.io/gh/EpiAware/ReparameterisedDistributions.jl?flags%5B0%5D=ad-enzyme-forward) | [![cov Enzyme reverse](https://codecov.io/gh/EpiAware/ReparameterisedDistributions.jl/graph/badge.svg?flag=ad-enzyme-reverse)](https://app.codecov.io/gh/EpiAware/ReparameterisedDistributions.jl?flags%5B0%5D=ad-enzyme-reverse) | [![cov Mooncake reverse](https://codecov.io/gh/EpiAware/ReparameterisedDistributions.jl/graph/badge.svg?flag=ad-mooncake-reverse)](https://app.codecov.io/gh/EpiAware/ReparameterisedDistributions.jl?flags%5B0%5D=ad-mooncake-reverse) | [![cov Mooncake forward](https://codecov.io/gh/EpiAware/ReparameterisedDistributions.jl/graph/badge.svg?flag=ad-mooncake-forward)](https://app.codecov.io/gh/EpiAware/ReparameterisedDistributions.jl?flags%5B0%5D=ad-mooncake-forward) |
<!-- badges:end -->

*Parameter-convention switches for Distributions.jl.*

## Why ReparameterisedDistributions?

- A delay can be elicited as a mean and a standard deviation, but
  Distributions.jl names each family by its native parameters, so those are
  not the coordinates a model has to be written in.
- Independent priors on a shape and a scale imply a prior on the mean that
  was never chosen, and is usually not the one that was meant.
- `reparameterise` makes the moments a distribution's parameters, so a prior
  can be put on the mean directly.
- The wrapper is an ordinary `Distribution` and stays differentiable, so the
  moments can be sampled directly inside a model.

## Getting started

See the
[Getting started documentation](https://reparameteriseddistributions.epiaware.org/dev/getting-started/)
for every supported parameterisation.

```jl
using Pkg
Pkg.add("ReparameterisedDistributions")
```

`reparameterise` returns a distribution whose parameters *are* the moments, so
a prior goes on the mean rather than on a shape that only implies one.

```julia
using ReparameterisedDistributions, Distributions, Turing, Random

Random.seed!(1)

truth = reparameterise(Gamma; mean = 8.0, sd = 3.0)
y = rand(truth, 200)

@model function delay(y)
    delay_mean ~ truncated(Normal(8.0, 4.0); lower = 0.0)
    delay_sd ~ truncated(Normal(3.0, 2.0); lower = 0.0)
    y .~ reparameterise(Gamma; mean = delay_mean, sd = delay_sd)
end

chain = sample(delay(y), NUTS(), 500; progress = false)

Turing.FlexiChains.summarystats(chain)
```

The chain comes back in a mean and a standard deviation, the coordinates the
delay was elicited in, rather than in native parameters that only imply them.

```julia
using CairoMakie

CairoMakie.activate!(type = "png", px_per_unit = 2)

fig = Figure()
for (i, (label, draws, actual)) in enumerate((
        ("mean", vec(chain[:delay_mean]), 8.0),
        ("sd", vec(chain[:delay_sd]), 3.0)
    ))
    ax = Axis(fig[1, i]; title = label)
    density!(ax, draws)
    vlines!(ax, [actual]; color = :black, linestyle = :dash)
end
fig
```

## Related packages

- [ComposedDistributions.jl](https://composeddistributions.epiaware.org/dev/)
  builds delay distributions by composing leaves; a reparameterised leaf
  carries its moments as its parameters, so a prior sits on a component's mean
  rather than on a native shape.
- [ConvolvedDistributions.jl](https://convolveddistributions.epiaware.org/dev/),
  [ModifiedDistributions.jl](https://modifieddistributions.epiaware.org/dev/)
  and
  [CensoredDistributions.jl](https://censoreddistributions.epiaware.org/stable/)
  build the convolved, transformed and censored distributions this package can
  wrap in moment coordinates for fitting.
- [DistributionsInference.jl](https://github.com/EpiAware/DistributionsInference.jl)
  is the emerging fit-protocol layer across the EpiAware distribution packages,
  where reparameterising to estimable moments is most useful.
- [Distributions.jl](https://juliastats.org/Distributions.jl/stable/) supplies
  the native families whose parameterisation this package switches.

## Where to learn more

- [Getting started](https://reparameteriseddistributions.epiaware.org/dev/getting-started/),
  for the full walkthrough.
- [Documentation](https://reparameteriseddistributions.epiaware.org/dev/)
- [EpiAware](https://github.com/EpiAware), the wider ecosystem this package
  belongs to.

<!-- standard-sections:start -->
<!-- MANAGED by EpiAwarePackageTools.scaffold — do not edit between the
     markers. These standard sections are re-rendered on every update;
     edit the package-owned sections outside them, or CITATION.cff. -->

## Contributing

We welcome contributions and new contributors! Please open an issue or pull request on [GitHub](https://github.com/EpiAware/ReparameterisedDistributions.jl). This package follows [ColPrac](https://github.com/SciML/ColPrac) and is formatted with [Runic](https://github.com/fredrikekre/Runic.jl).

## How to cite

If you use ReparameterisedDistributions in your work, please cite it. Citation metadata lives in [`CITATION.cff`](https://github.com/EpiAware/ReparameterisedDistributions.jl/blob/main/CITATION.cff), which GitHub renders as a "Cite this repository" button on the repository page.

## Code of conduct

Please note that the ReparameterisedDistributions project is released with a [Contributor Code of Conduct](https://github.com/EpiAware/.github/blob/main/CODE_OF_CONDUCT.md). By contributing, you agree to abide by its terms.
<!-- standard-sections:end -->
