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
are ignored — only its family is taken. The keywords are order-insensitive, as
keywords are everywhere else.

# Arguments
- `dist_or_type`: the native family to wrap, as a type or an instance.
- `check_args`: whether to reject invalid parameters at construction. Left on by
  default. A sampler exploring an unconstrained parameter turns it off: an
  invalid proposal then gives `logpdf == -Inf` (and `pdf == 0`) rather than an
  exception raised in the middle of a gradient. Every other method still
  converts, so an invalid distribution has no mean, no quantile and no draw, and
  asking for one raises.
- `alt_params`: the alternative parameters, as keywords.

!!! note
    `params` reports the moments, so the usual
    `typeof(d)(params(d)...)` idiom does not rebuild one of these — the family
    and the parameter names live in type parameters. Rebuild through
    `reparameterise` instead. Generic code relying on that idiom will raise
    rather than silently misbehave.

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
    return _build(D, keys(nt), Tuple(nt); check_args = check_args)
end

function reparameterise(d::Distribution; kwargs...)
    return reparameterise(Base.typename(typeof(d)).wrapper; kwargs...)
end

# Keyword arguments are order-insensitive everywhere else in Julia, but `keys` of
# a keyword NamedTuple preserves the CALL-SITE order and the conversions dispatch
# on those names. Sort into a canonical order so that
# `reparameterise(LogNormal; sd = 2.0, mean = 8.0)` means what
# `reparameterise(LogNormal; mean = 8.0, sd = 2.0)` means.
#
# Generated, so the sort happens once at compile time from the names alone and
# the emitted code is a bare tuple permutation. Sorting at run time would compare
# `Symbol`s, and comparing `Symbol`s goes through a `ccall` (`jl_symbol_name`)
# that Mooncake cannot differentiate — and this sits on the sampler's hot path,
# because a model reconstructs the distribution at every gradient evaluation.
@generated function _canonical(::Val{names}, vals::Tuple) where {names}
    p = sortperm(collect(names))
    sorted = Tuple(collect(names)[p])
    permuted = Expr(:tuple, (:(vals[$(p[i])]) for i in eachindex(p))...)
    return :(($(QuoteNode(sorted)), $permuted))
end

# Shared construction path: `reparameterise` is the front door, the ecosystem's
# leaf-rebuild hook calls this with the names already fixed. Both go through
# here, so both canonicalise and promote alike — a leaf rebuilt from an
# `Int`/`Float64` mix must not end up with an abstract `NTuple{2, Real}` field,
# which would be boxed, type-unstable and hostile to a gradient.
function _build(::Type{D}, names::Tuple{Vararg{Symbol}},
        vals::Tuple{Vararg{Real}}; check_args::Bool = true) where {D}
    length(names) == length(vals) || throw(ArgumentError(
        "expected one value per parameter name, got $(length(names)) names " *
        "and $(length(vals)) values"))
    cnames, cvals = _canonical(Val(names), vals)
    pvals = promote(map(float, cvals)...)
    d = _reparameterised(D, cnames, pvals)
    if check_args
        _check_moments(D, Val(cnames), pvals)
        _check_native(d)
    end
    return d
end

@doc raw"

Whether a family's alternative parameters are valid, answered without throwing.

This is the predicate [`_check_moments`](@ref) throws on, and — the reason it is
separate — the predicate the density guards with. A sampler exploring an
unconstrained parameter will propose an invalid point; it needs `-Inf` back, not
an exception raised in the middle of a gradient. So `logpdf` consults this and
returns `-Inf` rather than converting to a native distribution that would either
throw or, worse, be silently valid.

Silently valid is the real hazard. The LogNormal and Gamma conversions square the
standard deviation, so a negative one maps onto exactly the same native
distribution as its positive counterpart: without this predicate the density
would be finite, and identical to the density at `+sd`, giving a mirror mode in
an unconstrained parameterisation. Checking at construction alone does not help,
because that check is precisely what a sampler turns off.

The fallback accepts anything; a family adds a method alongside its
[`_to_native`](@ref).

# Arguments
- the native family being checked for.
- `Val(names)`: the alternative parameter names.
- `vals`: the alternative parameter values, in `names` order.
"
_valid_moments(::Type{D}, ::Val{names}, vals) where {D, names} = true

# Whether this wrapper's own moments are valid.
function _valid(d::Reparameterised{D, names}) where {D, names}
    return _valid_moments(D, Val(names), d.vals)
end

@doc raw"

Check that a family's alternative parameters are themselves valid, throwing if
they are not.

Checking the native distribution is not enough. A closed form can map an invalid
moment onto a perfectly valid native distribution: a negative standard deviation
squares away in the LogNormal conversion, yielding the same native distribution
as its positive counterpart, so the wrapper would report a parameter it does not
behave as. The moments have to be checked in their own coordinates.

A family states its constraints once, in [`_valid_moments`](@ref); this raises on
them. The message is per-family so that a caller learns which moment was wrong.

# Arguments
- the native family being checked for.
- `Val(names)`: the alternative parameter names.
- `vals`: the alternative parameter values, in `names` order.
"
function _check_moments(::Type{D}, ::Val{names}, vals) where {D, names}
    _valid_moments(D, Val(names), vals) || throw(DomainError(vals,
        "invalid $(collect(names)) for $(D)"))
    return nothing
end

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

# The type a density must come back as, so that `-Inf` at an invalid point is a
# `Dual` under AD rather than a bare `Float64` that would break the tape.
function _restype(d::Reparameterised, x::Real)
    return promote_type(eltype(d.vals), typeof(float(x)))
end

# The two density methods are the sampler's hot path, and the only ones that
# guard: an invalid point yields `-Inf` (a zero density) rather than an error
# raised mid-gradient, which is the whole point of `check_args = false`. Every
# other method converts, so an invalid distribution has no mean, no quantile and
# no draw — asking for one raises, which is the honest answer.
function logpdf(d::Reparameterised, x::Real)
    _valid(d) || return convert(_restype(d, x), -Inf)
    return logpdf(_native(d), x)
end

function pdf(d::Reparameterised, x::Real)
    _valid(d) || return zero(_restype(d, x))
    return pdf(_native(d), x)
end

cdf(d::Reparameterised, x::Real) = cdf(_native(d), x)
logcdf(d::Reparameterised, x::Real) = logcdf(_native(d), x)
ccdf(d::Reparameterised, x::Real) = ccdf(_native(d), x)
logccdf(d::Reparameterised, x::Real) = logccdf(_native(d), x)
quantile(d::Reparameterised, q::Real) = quantile(_native(d), q)

mean(d::Reparameterised) = mean(_native(d))
var(d::Reparameterised) = var(_native(d))
# `std` and `median` fall out of `var` and `quantile`, but these do not, and
# without them they reach a Base generic and fail with an opaque `iterate` error
# rather than doing the obvious thing. A package sold on moments should report
# its moments.
mode(d::Reparameterised) = mode(_native(d))
modes(d::Reparameterised) = modes(_native(d))
skewness(d::Reparameterised) = skewness(_native(d))
kurtosis(d::Reparameterised) = kurtosis(_native(d))
entropy(d::Reparameterised) = entropy(_native(d))

sampler(d::Reparameterised) = sampler(_native(d))
Base.rand(rng::AbstractRNG, d::Reparameterised) = rand(rng, _native(d))

function Base.show(io::IO, d::Reparameterised{D, names}) where {D, names}
    args = join(("$n = $v" for (n, v) in zip(names, d.vals)), ", ")
    return print(io, "reparameterise(", D, "; ", args, ")")
end
