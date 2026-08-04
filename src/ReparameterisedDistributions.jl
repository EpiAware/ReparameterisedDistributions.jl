"""
    ReparameterisedDistributions

Parameter-convention switches for Distributions.jl: wrap a distribution so that
it is parameterised by the quantities a modeller reasons about — its moments —
rather than by its native parameters.

Distributions.jl parameterises each family by its native parameters: a `Gamma`
by shape and scale, a `LogNormal` by the mean and standard deviation of its
logarithm. A delay distribution, though, is elicited as a mean and a standard
deviation, and a prior belongs on the mean. Such a prior cannot be expressed
through a native leaf, because independent priors on shape and scale do not
compose into a prior on the mean.

[`reparameterise`](@ref) returns a `Distribution` whose parameters *are* the
moments. It evaluates and samples exactly as the native distribution does, so it
can be used directly on the left of a `~`; it converts to the native family
through an exact closed form; and it stays differentiable, so the moments can be
sampled. [`native`](@ref) reaches the native distribution — and, through it,
the native parameters — when the moments alone are not enough. [`rescale`](@ref)
scales a registered moment while holding the others fixed, routing through the
same closed form.

# Examples
```@example
using ReparameterisedDistributions, Distributions

d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
(params(d), mean(d), std(d))
```
"""
module ReparameterisedDistributions

using Random: AbstractRNG
# Docstring-template machinery used by src/docstrings.jl (imports are
# centralised here per the kit's import-centralisation gate).
using DocStringExtensions: @template, DOCSTRING, EXPORTS, IMPORTS, TYPEDEF,
                           TYPEDFIELDS, TYPEDSIGNATURES
# The Weibull numeric conversion's own mathematics (src/families.jl):
# already in this package's dependency closure through Distributions, so
# this costs no new install and no new transitive package.
using SpecialFunctions: digamma, loggamma

# Functions we extend for the wrapper.
import Distributions: params, insupport, pdf, logpdf, cdf, logcdf, ccdf,
                      logccdf, quantile, mean, var, sampler, mode, modes,
                      skewness, kurtosis, entropy, mgf, cf, loglikelihood
# Types and constructors we use without extending.
using Distributions: Distributions, Distribution, Beta, Exponential, Gamma,
                     InverseGaussian, LogNormal, NegativeBinomial, SkewNormal,
                     Univariate, VariateForm, ValueSupport, Weibull

# Register the standard EpiAware docstring conventions before any docstrings
# are defined (see src/docstrings.jl).
include("docstrings.jl")

# The verbs, and the two conversion functions, are exported; the wrapper type
# and its supertype are public but not exported (see public.jl), following
# the ecosystem convention.
export reparameterise, rescale, to_native, native

# The abstract supertype, then the concrete wrapper and its front door, then
# the numeric fallback the front door dispatches to when no closed form is
# registered, then the per-family conversions themselves (closed-form and
# numeric alike).
include("interface.jl")
include("Reparameterised.jl")
include("numeric.jl")
include("families.jl")

@static if VERSION >= v"1.11"
    include("public.jl")
end

end # module ReparameterisedDistributions
