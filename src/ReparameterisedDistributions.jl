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
sampled.

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

# Functions we extend for the wrapper.
import Distributions: params, insupport, pdf, logpdf, cdf, logcdf, ccdf,
                      logccdf, quantile, mean, var, sampler
# Types and constructors we use without extending.
using Distributions: Distributions, Distribution, LogNormal, VariateForm,
                     ValueSupport

# Register the standard EpiAware docstring conventions before any docstrings
# are defined (see src/docstrings.jl).
include("docstrings.jl")

# The verb is exported; the wrapper type and its supertype are public but not
# exported (see public.jl), following the ecosystem convention.
export reparameterise

# The abstract supertype, then the concrete wrapper and its front door, then the
# per-family closed-form conversions it dispatches to.
include("interface.jl")
include("Reparameterised.jl")
include("families.jl")

@static if VERSION >= v"1.11"
    include("public.jl")
end

end # module ReparameterisedDistributions
