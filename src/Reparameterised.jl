@doc raw"

A native Distributions.jl family stood up under an alternative parameterisation.

Stores the alternative parameter values in the order of the registered `names`,
and converts to the native family through [`_to_native`](@ref) whenever a
density, moment or sample is asked for. `params` reports the alternative values,
not the native ones, so it is those that the ecosystem's parameter introspection
sees and that a prior can be placed on.

The family `D`, the parameter `names` and the variate form and value support are
type parameters, so a wrapper around a discrete family stays discrete and the
conversion is resolved at compile time.

# Fields
- `vals`: the alternative parameter values, in registered `names` order.

# See also
- [`reparameterise`](@ref): the public constructor.
"
struct Reparameterised{D, names, N, T <: Real, F <: VariateForm,
    S <: ValueSupport} <: AbstractReparameterisedDistribution{F, S}
    "The alternative parameter values, in registered `names` order."
    vals::NTuple{N, T}
end

# Build the wrapper, taking the variate form and value support from the family
# being wrapped so a discrete family (a NegativeBinomial by mean and
# overdispersion, say) does not silently become continuous.
function _reparameterised(::Type{D}, names::Tuple{Vararg{Symbol}},
        vals::Tuple{Vararg{Real}}) where {D}
    F = Distributions.variate_form(D)
    S = Distributions.value_support(D)
    return Reparameterised{D, names, length(vals), eltype(vals), F, S}(vals)
end

@doc raw"

Wrap `dist_or_type` so that the keyword parameters given here are its parameters.

The keywords name an alternative parameterisation of the family, and the wrapper
converts to the native family internally through an exact closed form. The
result is a `Distribution`, so it evaluates and samples exactly as the native
distribution does and can be used directly on the left of a `~` in a
probabilistic model.

The point of the wrapper is that the alternative parameters remain *the*
parameters: `params` reports them, and the ecosystem's parameter introspection
places a prior on them, rather than on the native parameters that merely imply
them. A prior on a delay's mean cannot be expressed through a native
`Gamma(shape, scale)` leaf, because independent priors on `shape` and `scale` do
not compose into a prior on the mean.

Pass either the family (`LogNormal`) or an instance of it, whose parameter values
are ignored — only its family is taken.

# Arguments
- `dist_or_type`: the native family to wrap, as a type or an instance.
- `check_args`: whether to check that the parameters imply a valid native
  distribution. Left on by default; a sampler exploring an invalid point should
  turn it off so the density returns `-Inf` rather than throwing mid-gradient.
- `alt_params`: the alternative parameters, as keywords.

# Examples
```@example
using ReparameterisedDistributions, Distributions

d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
params(d)
```

```@example
using ReparameterisedDistributions, Distributions

mean(reparameterise(LogNormal; mean = 8.0, sd = 2.0))
```
"
function reparameterise(::Type{D}; check_args::Bool = true,
        alt_params...) where {D <: Distribution}
    nt = values(alt_params)
    isempty(nt) && throw(ArgumentError(
        "reparameterise($(D)) needs the alternative parameters as keywords, " *
        "e.g. reparameterise($(D); mean = 8.0, sd = 2.0)"))
    vals = promote(map(float, Tuple(nt))...)
    return _build(D, keys(nt), vals; check_args = check_args)
end

function reparameterise(d::Distribution; kwargs...)
    return reparameterise(Base.typename(typeof(d)).wrapper; kwargs...)
end

# Shared construction path: `reparameterise` is the front door, the ecosystem's
# leaf-rebuild hook calls this with the names already fixed.
function _build(::Type{D}, names::Tuple{Vararg{Symbol}},
        vals::Tuple{Vararg{Real}}; check_args::Bool = true) where {D}
    length(names) == length(vals) || throw(ArgumentError(
        "expected one value per parameter name, got $(length(names)) names " *
        "and $(length(vals)) values"))
    d = _reparameterised(D, names, vals)
    if check_args
        _check_moments(D, Val(names), vals)
        _check_native(d)
    end
    return d
end

@doc raw"

Check that a family's alternative parameters are themselves valid.

Checking the native distribution is not enough. A closed form can map an invalid
moment onto a perfectly valid native distribution: a negative standard deviation
squares away in the LogNormal conversion, yielding the same native distribution
as its positive counterpart, so the wrapper would report a parameter it does not
behave as. The moments have to be checked in their own coordinates.

The fallback accepts anything; a family adds a method alongside its
[`_to_native`](@ref).

# Arguments
- the native family being checked for.
- `Val(names)`: the alternative parameter names.
- `vals`: the alternative parameter values, in `names` order.
"
_check_moments(::Type{D}, ::Val{names}, vals) where {D, names} = nothing

# Force the native conversion through the family's own argument checks once, at
# construction. `_to_native` itself builds with `check_args = false` so it stays
# branch-free and differentiable on the hot path.
function _check_native(d::Reparameterised)
    native = _native(d)
    Base.typename(typeof(native)).wrapper(Distributions.params(native)...)
    return nothing
end

@doc raw"

Convert a wrapper's alternative parameters to the native distribution.

The per-family closed forms are the methods of [`_to_native`](@ref); this is the
dispatch point every density, moment and sampling method goes through.
"
function _native(d::Reparameterised{D, names}) where {D, names}
    return _to_native(D, Val(names), d.vals)
end

@doc raw"

The closed-form conversion from a family's alternative parameters to the native
distribution.

Each supported (family, parameter-name) pair adds a method. A method must be
exact algebra rather than a numerical solve, and must build the native
distribution with `check_args = false`, so the conversion stays differentiable
and a sampler probing an invalid point yields `-Inf` rather than throwing
mid-gradient.

# Arguments
- the native family being converted to.
- `Val(names)`: the alternative parameter names, as a value type so the
  conversion is resolved at compile time.
- `vals`: the alternative parameter values, in `names` order.

# Examples
```@example
using ReparameterisedDistributions, Distributions

ReparameterisedDistributions._to_native(
    LogNormal, Val((:mean, :sd)), (8.0, 2.0))
```

# See also
- [`reparameterise`](@ref): the public constructor that dispatches to this.
"
function _to_native(::Type{D}, ::Val{names}, vals) where {D, names}
    throw(ArgumentError(
        "no reparameterisation of $(D) by $(collect(names)) is registered; " *
        "the registered parameterisations are listed in the package docs"))
end

# The alternative parameter names this wrapper was built with.
_names(::Reparameterised{D, names}) where {D, names} = names

# --- Distributions.jl interface --------------------------------------------
#
# The moments are the parameters: `params` reports the alternative values, so
# the ecosystem's parameter introspection reads and rebuilds in those
# coordinates. Everything else delegates to the native distribution.

params(d::Reparameterised) = d.vals

Base.minimum(d::Reparameterised) = minimum(_native(d))
Base.maximum(d::Reparameterised) = maximum(_native(d))

insupport(d::Reparameterised, x::Real) = insupport(_native(d), x)

pdf(d::Reparameterised, x::Real) = pdf(_native(d), x)
logpdf(d::Reparameterised, x::Real) = logpdf(_native(d), x)
cdf(d::Reparameterised, x::Real) = cdf(_native(d), x)
logcdf(d::Reparameterised, x::Real) = logcdf(_native(d), x)
ccdf(d::Reparameterised, x::Real) = ccdf(_native(d), x)
logccdf(d::Reparameterised, x::Real) = logccdf(_native(d), x)
quantile(d::Reparameterised, q::Real) = quantile(_native(d), q)

mean(d::Reparameterised) = mean(_native(d))
var(d::Reparameterised) = var(_native(d))

sampler(d::Reparameterised) = sampler(_native(d))
Base.rand(rng::AbstractRNG, d::Reparameterised) = rand(rng, _native(d))

function Base.show(io::IO, d::Reparameterised{D, names}) where {D, names}
    args = join(("$n = $v" for (n, v) in zip(names, d.vals)), ", ")
    return print(io, "reparameterise(", D, "; ", args, ")")
end
