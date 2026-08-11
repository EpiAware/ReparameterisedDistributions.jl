@doc raw"

A native distribution family stood up under an alternative parameterisation.

Stores the alternative parameter values in the order of the registered `names`,
and converts to the native family through [`to_native`](@ref) whenever a
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
function _reparameterised(::Type{D}, names::Tuple{Vararg{Union{Symbol, Real}}},
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

An elicitation often arrives as quantiles rather than moments: a delay quoted
as a 5th and a 95th percentile. The `quantiles` keyword takes
`probability => value` pairs, and the elicited values become the parameters, so
a prior goes on the quantity that was elicited. Probabilities are arbitrary, and
quantiles mix with moment keywords — one moment and one tail point is a common
elicitation. The number of constraints has to match the family's native
parameter count.

# Arguments
- `dist_or_type`: the native family to wrap, as a type or an instance.
- `check_args`: whether to reject invalid parameters at construction. Left on by
  default. A sampler exploring an unconstrained parameter turns it off: an
  invalid proposal then gives `logpdf == -Inf` (and `pdf == 0`) rather than an
  exception raised in the middle of a gradient. Every other method still
  converts, so an invalid distribution has no mean, no quantile and no draw, and
  asking for one raises.
- `alt_params`: the alternative parameters, as keywords. `quantiles` is
  reserved for elicited quantiles, given as `probability => value` pairs.

!!! note
    `params` reports the moments, so the usual
    `typeof(d)(params(d)...)` idiom does not rebuild one of these — the family
    and the parameter names live in type parameters. Rebuild through
    `reparameterise` instead. Generic code relying on that idiom will raise
    rather than silently misbehave. For the native parameters — the ones the
    wrapped family was actually built from — use `params(native(d))`.

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

```@example
using ReparameterisedDistributions, Distributions

d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
params(native(d))
```

```@example
using ReparameterisedDistributions, Distributions

d = reparameterise(LogNormal; quantiles = (0.05 => 1.2, 0.95 => 8.4))
(params(d), quantile(d, 0.05), quantile(d, 0.95))
```

```@example
using ReparameterisedDistributions, Distributions

reparameterise(LogNormal; median = 4.0, quantiles = (0.95 => 12.0,))
```

# See also
- [`native`](@ref): the native distribution a wrapper converts to.
"
function reparameterise(::Type{D}; check_args::Bool = true,
        alt_params...) where {D <: Distribution}
    nt = values(alt_params)
    isempty(nt) && throw(ArgumentError(
        "reparameterise($(D)) needs the alternative parameters as keywords, " *
        "e.g. reparameterise($(D); mean = 8.0, sd = 2.0)"))
    # `haskey` on a `NamedTuple` reads its type, so the branch is settled at
    # compile time and only one path is emitted.
    haskey(nt, :quantiles) &&
        return _build_quantiles(D, nt; check_args = check_args)
    return _build(D, Val(keys(nt)), Tuple(nt); check_args = check_args)
end

function reparameterise(d::Distribution; kwargs...)
    return reparameterise(Base.typename(typeof(d)).wrapper; kwargs...)
end

@doc raw"

Scale `d`'s named parameter by `factor`, holding the others fixed.

Routes through whichever moment parameterisation `d` was itself built under —
the registered `names` its type already carries — so scaling a mean (or any
other registered parameter) has one home rather than a caller hand-rebuilding
the wrapper from `params(d)`. An affine transform is not a substitute: for a
discrete family such as `NegativeBinomial`, scaling the native support does not
scale the mean cleanly, so the scaling has to happen in moment coordinates and
convert back through the family's own closed form.

`parameter` must be one of `d`'s registered names, or this throws a
`DomainError` rather than silently applying the factor under different
semantics. `d` must itself be a [`Reparameterised`](@ref) distribution — a
native, unwrapped `Distribution` has no registered parameterisation to route
through and is rejected with an `ArgumentError` naming the family to wrap
first.

# Arguments
- `d`: the distribution to rescale.
- `factor`: the multiplicative factor applied to `parameter`.
- `parameter`: the registered name to scale. Defaults to `:mean`.
- `check_args`: forwarded to the rebuilt wrapper; see [`reparameterise`](@ref).

# Examples
```@example
using ReparameterisedDistributions, Distributions

d = reparameterise(Gamma; mean = 8.0, shape = 2.0)
mean(rescale(d, 2.0))
```

# See also
- [`reparameterise`](@ref): the constructor `rescale` rebuilds through.
"
function rescale(d::Reparameterised{D, names}, factor::Real;
        parameter::Symbol = :mean, check_args::Bool = true) where {D, names}
    idx = findfirst(==(parameter), names)
    idx === nothing && throw(DomainError(parameter,
        "$(D) is not registered by a `$(parameter)` parameter; the " *
        "registered parameters are $(names)"))
    scaled = ntuple(i -> i == idx ? d.vals[i] * factor : d.vals[i],
        length(names))
    return _build(D, Val(names), scaled; check_args = check_args)
end

function rescale(d::Distribution, factor::Real; parameter::Symbol = :mean,
        check_args::Bool = true)
    throw(ArgumentError(
        "rescale needs a distribution built by `reparameterise`, which " *
        "fixes the registered parameterisation to route through; wrap " *
        "$(Base.typename(typeof(d)).wrapper) first, e.g. " *
        "rescale(reparameterise($(Base.typename(typeof(d)).wrapper); " *
        "mean = ..., ...), $(factor))"))
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
#
# A name is a `Symbol` for a moment and a probability for an elicited
# quantile, so the order is over a heterogeneous tuple: moments first,
# alphabetically, then quantiles by ascending probability.
_name_order(n::Symbol) = (0, string(n), 0.0)
_name_order(n::Real) = (1, "", Float64(n))

@generated function _canonical(::Val{names}, vals::Tuple) where {names}
    p = sortperm(collect(names), by = _name_order)
    sorted = Tuple(collect(names)[p])
    permuted = Expr(:tuple, (:(vals[$(p[i])]) for i in eachindex(p))...)
    return :(($(QuoteNode(sorted)), $permuted))
end

# Shared construction path: `reparameterise` is the front door, the ecosystem's
# leaf-rebuild hook calls this with the names already fixed. Both go through
# here, so both canonicalise and promote alike — a leaf rebuilt from an
# `Int`/`Float64` mix must not end up with an abstract `NTuple{2, Real}` field,
# which would be boxed, type-unstable and hostile to a gradient.
#
# `names` arrives as a `Val`, not a bare `Tuple{Vararg{Symbol}}`, and that is
# not stylistic: `Val(runtime_tuple)` cannot be inferred concretely from a
# plain tuple argument — the value only becomes a type parameter if the
# CALLER already carried it as one (which `reparameterise` does, via the
# kwcall specialising on the keyword names) and passes it through as `Val`.
# Accepting the bare tuple here, and calling `Val(names)` on it inside the
# function body, was exactly that mistake: `Reparameterised`'s own `names`,
# `N` and `T` type parameters came back uninferred (`@inferred` on
# `reparameterise` failed) for any call that was not fully constant-folded —
# a real risk under AD, where the surrounding tape rarely preserves the
# constant propagation a bare literal call gets at the top level.
function _build(::Type{D}, ::Val{names},
        vals::Tuple{Vararg{Real}}; check_args::Bool = true) where {D, names}
    length(names) == length(vals) || throw(ArgumentError(
        "expected one value per parameter name, got $(length(names)) names " *
        "and $(length(vals)) values"))
    cnames, cvals = _canonical(Val(names), vals)
    pvals = promote(map(float, cvals)...)
    d = _reparameterised(D, cnames, pvals)
    if check_args
        valid_moments(D, Val(cnames), pvals) || throw(DomainError(pvals,
            "invalid $(collect(cnames)) for $(D)"))
        _check_native(to_native(D, Val(cnames), pvals))
    end
    return d
end

# The docstring below must stay immediately adjacent to the function it
# documents: even a comment between them silently detaches it (verified —
# Aqua's undocumented-names check is what catches this).
@doc raw"

Whether a family's alternative parameters describe a member of the family,
answered without converting or throwing.

Checked BEFORE [`to_native`](@ref) at every call site, hot path and
construction alike — a family's own `to_native` method assumes its input has
already passed this check, and is free to build whatever it builds without
guarding, so it always returns a concrete distribution rather than a
`Union` of one and `nothing`. Keeping that return type concrete is not
cosmetic: an AD-hot call site that instead bound a
`Union{Nothing, <native type>}`-typed conversion result has produced a
silently wrong reverse-mode gradient in this package before, with no error
and no warning; a family's own math never has to defend against that on
its own so long as it registers a truthful predicate here.

`nothing` cannot be recovered from the native distribution's own type: some
conversions are even in the invalid direction (the LogNormal and Gamma
conversions square the standard deviation, so a negative one builds exactly
the same, perfectly valid native distribution as its positive counterpart),
so the check has to happen in the alternative parameters' own coordinates,
in a method registered here, before [`to_native`](@ref) runs.

Each supported (family, parameter-name) pair adds a method alongside its
[`to_native`](@ref) registration. The 3-arg fallback accepts anything, so a
family that registers `to_native` without a matching `valid_moments` method
is silently treated as always valid: on the `check_args = false` hot path
nothing surfaces the omission, and `logpdf`/`pdf`/`loglikelihood` return a
finite, wrong density instead of `-Inf` at an invalid point. Register both
methods together.

# Arguments
- the native family being checked for.
- `Val(names)`: the alternative parameter names, as a value type so the
  check is resolved at compile time.
- `vals`: the alternative parameter values, in `names` order.

# Returns
`Bool`. Must not throw, and must also exclude points [`to_native`](@ref)
cannot represent even though the moment itself looks fine, such as a
numeric family's solvable window.

# Examples
```@example
using ReparameterisedDistributions, Distributions

valid_moments(LogNormal, Val((:mean, :sd)), (8.0, 2.0))
```

```@example
using ReparameterisedDistributions, Distributions

valid_moments(LogNormal, Val((:mean, :sd)), (8.0, -1.0))
```

# See also
- [`to_native`](@ref): the conversion this guards, registered alongside it.
"
valid_moments(::Type{D}, ::Val{names}, vals) where {D, names} = true

# Force the native conversion through the family's own argument checks once,
# at construction. `to_native` itself builds with `check_args = false` so it
# stays branch-free and differentiable on the hot path; this catches a class
# of invalid input the moment guard structurally cannot, such as an infinite
# moment (`Gamma(mean = Inf, sd = 1.0)` passes `mean > 0` but is not a Gamma).
function _check_native(nd::Distribution)
    Base.typename(typeof(nd)).wrapper(Distributions.params(nd)...)
    return nothing
end

@doc raw"

The native distribution a wrapper's alternative parameters convert to.

Every density, moment and sampling method on a `Reparameterised` goes through
this, so it is also the way to reach the native parameters — the ones the
wrapped family was actually built from — when the moments alone are not
enough: `params(native(d))` rather than `params(d)`. Throws a `DomainError`
if `d`'s parameters are invalid — see [`valid_moments`](@ref).

# Examples
```@example
using ReparameterisedDistributions, Distributions

d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
native(d), params(native(d))
```

# See also
- [`to_native`](@ref): the per-family closed form this dispatches to.
- [`valid_moments`](@ref): the per-family guard checked first.
"
function native(d::Reparameterised{D, names}) where {D, names}
    valid_moments(D, Val(names), d.vals)::Bool || throw(DomainError(d.vals,
        "invalid $(collect(names)) for $(D)"))
    return to_native(D, Val(names), d.vals)
end

# A family's own `to_native` method — analytic or numeric, calling
# `solve_moment` inside its body — is always strictly more specific than
# this 3-arg fallback, so ordinary dispatch picks it first whenever one is
# registered; this generic method is only reached for an unregistered
# pair. The docstring below must stay immediately adjacent to the function
# it documents: even a comment between them silently detaches it (verified
# — Aqua's undocumented-names check is what catches this).
@doc raw"

The closed-form conversion from a family's alternative parameters to the
native distribution.

Each supported (family, parameter-name) pair adds a method. A method should
be exact algebra where a closed form exists, and otherwise call
[`solve_moment`](@ref). It must build the native distribution with
`check_args = false`, so the conversion stays differentiable and a sampler
probing an invalid point yields `-Inf` rather than throwing mid-gradient.

A method does NOT need to guard its own input: [`valid_moments`](@ref) is
checked at every call site first, so this always runs on parameters already
known valid and always returns a concrete `D`, never a `Union` of one and
`nothing`. Keep it that way — do not add an early `return nothing` here.
That split, not a stylistic preference, is why this stays two methods
rather than one: an AD-hot call site that bound a `Union{Nothing, D}`-typed
conversion result has produced a silently wrong reverse-mode gradient in
this package before, and a call site here can only avoid that failure mode
if every registered `to_native` method is unconditionally concrete.

Calling this directly, rather than through [`native`](@ref) or a wrapper's
own methods, without first checking [`valid_moments`](@ref) is undefined:
it may return a meaningless distribution or throw, never `nothing`.

# Arguments
- the native family being converted to.
- `Val(names)`: the alternative parameter names, as a value type so the
  conversion is resolved at compile time.
- `vals`: the alternative parameter values, in `names` order.

# Examples
```@example
using ReparameterisedDistributions, Distributions

to_native(LogNormal, Val((:mean, :sd)), (8.0, 2.0))
```

# See also
- [`reparameterise`](@ref): the public constructor that dispatches to this.
- [`native`](@ref): the wrapper-level accessor most callers want instead.
- [`valid_moments`](@ref): the guard checked before this runs.
"
function to_native(::Type{D}, ::Val{names}, vals) where {D, names}
    throw(ArgumentError(
        "no reparameterisation of $(D) by $(collect(names)) is " *
        "registered; the registered parameterisations are listed " *
        "in the package docs"))
end

# --- Distributions.jl interface --------------------------------------------
#
# The moments are the parameters: `params` reports the alternative values, so
# the ecosystem's parameter introspection reads and rebuilds in those
# coordinates. Everything else delegates to the native distribution.

params(d::Reparameterised) = d.vals

Base.minimum(d::Reparameterised) = minimum(native(d))
Base.maximum(d::Reparameterised) = maximum(native(d))

insupport(d::Reparameterised, x::Real) = insupport(native(d), x)

# The type a density must come back as, so that `-Inf` at an invalid point is a
# `Dual` under AD rather than a bare `Float64` that would break the tape.
function _restype(d::Reparameterised, x::Real)
    return promote_type(eltype(d.vals), typeof(float(x)))
end

# The density methods (and `loglikelihood`, below) are the sampler's hot
# path, and the only ones that guard: an invalid point yields `-Inf` (a zero
# density) rather than an error raised mid-gradient, which is the whole point
# of `check_args = false`. Every other method converts through `native`, so
# an invalid distribution has no mean, no quantile and no draw — asking for
# one raises, which is the honest answer. These check `valid_moments` and
# call `to_native` directly rather than `native`, so an invalid point never
# routes through a throw.
#
# `valid_moments` is checked and branched on BEFORE `to_native` runs, never
# after: `nd` below is always a concretely-typed `D`, never bound as
# `Union{Nothing, D}`. Do not collapse this back into a single call that
# branches on `to_native`'s result — see `to_native`'s own docstring for why.
function logpdf(d::Reparameterised{D, names}, x::Real) where {D, names}
    valid_moments(D, Val(names), d.vals)::Bool ||
        return convert(_restype(d, x), -Inf)
    nd = to_native(D, Val(names), d.vals)
    return logpdf(nd, x)
end

function pdf(d::Reparameterised{D, names}, x::Real) where {D, names}
    valid_moments(D, Val(names), d.vals)::Bool || return zero(_restype(d, x))
    nd = to_native(D, Val(names), d.vals)
    return pdf(nd, x)
end

cdf(d::Reparameterised, x::Real) = cdf(native(d), x)
logcdf(d::Reparameterised, x::Real) = logcdf(native(d), x)
ccdf(d::Reparameterised, x::Real) = ccdf(native(d), x)
logccdf(d::Reparameterised, x::Real) = logccdf(native(d), x)
quantile(d::Reparameterised, q::Real) = quantile(native(d), q)

mean(d::Reparameterised) = mean(native(d))
var(d::Reparameterised) = var(native(d))
# `std` and `median` fall out of `var` and `quantile`, but these do not, and
# without them they reach a Base generic and fail with an opaque `iterate` error
# rather than doing the obvious thing. A package sold on moments should report
# its moments.
mode(d::Reparameterised) = mode(native(d))
modes(d::Reparameterised) = modes(native(d))
skewness(d::Reparameterised) = skewness(native(d))
kurtosis(d::Reparameterised) = kurtosis(native(d))
entropy(d::Reparameterised) = entropy(native(d))
mgf(d::Reparameterised, t::Real) = mgf(native(d), t)
cf(d::Reparameterised, t::Real) = cf(native(d), t)

sampler(d::Reparameterised) = sampler(native(d))
Base.rand(rng::AbstractRNG, d::Reparameterised) = rand(rng, native(d))

# Distributions.jl's generic `loglikelihood` sums scalar `logpdf` calls, so
# without this a wrapper re-converts to its native distribution once per
# observation. That is cheap for an analytical conversion but costly for a
# numeric one, which re-runs a root-find: converting once and delegating is a
# 6-9x speed-up on a batch, and is exact either way since the native
# distribution does not depend on `x`.
#
# Guards the same way `logpdf` does: an invalid point gives `-Inf` rather
# than either throwing (via `native`) or silently bypassing the guard
# entirely, which the equivalent method did before this check existed.
#
# Restricted to `F = Univariate` (every family registered today) rather
# than left generic over `F`, because `Univariate` is itself a type alias
# for `ArrayLikeVariate{0}`: a fully generic method here would be
# ambiguous with Distributions' own
# `loglikelihood(d::Distribution{ArrayLikeVariate{N}}, x::AbstractArray)`
# for that `N = 0` case, since neither signature is strictly more specific
# than the other on both arguments at once.
function loglikelihood(
        d::Reparameterised{D, names, N1, T, Univariate},
        x::AbstractArray{<:Real, M}) where {D, names, N1, T, M}
    valid_moments(D, Val(names), d.vals)::Bool ||
        return convert(_restype(d, zero(eltype(x))), -Inf)
    nd = to_native(D, Val(names), d.vals)
    return loglikelihood(nd, x)
end

# The probabilities live in the type and `params` reports only the values,
# so the printed form has to put the pairs back together.
function _spec_string(names, vals)
    args = String["$n = $v" for (n, v) in zip(names, vals) if n isa Symbol]
    qs = String["$n => $v" for (n, v) in zip(names, vals) if !(n isa Symbol)]
    # A one-element tuple keeps its trailing comma, so the printed form is
    # still a tuple when pasted back.
    isempty(qs) ||
        push!(args, "quantiles = (" * join(qs, ", ") *
                    (length(qs) == 1 ? ",)" : ")"))
    return join(args, ", ")
end

function Base.show(io::IO, d::Reparameterised{D, names}) where {D, names}
    args = _spec_string(names, d.vals)
    # `nameof`, not `D` itself: printing the type directly qualifies it by
    # module whenever the active module differs from the one `D` is defined
    # in (a doctest sandbox, a TestItemRunner module, ...), so the same call
    # would print as `Distributions.Gamma` in one context and `Gamma` in
    # another. The short name is what a user actually typed and is always
    # in scope wherever `reparameterise` itself is, so it stays both
    # pasteable and stable across contexts.
    return print(io, "reparameterise(", nameof(D), "; ", args, ")")
end

# The compact form above stays code-reconstructable, so error messages and
# `join`/interpolation keep printing something a user can paste back in. The
# REPL can afford one more line: the native distribution the wrapper
# actually evaluates as, which is the thing a user most often wants to check
# when the moments are unfamiliar territory.
function Base.show(io::IO, ::MIME"text/plain",
        d::Reparameterised{D, names}) where {D, names}
    show(io, d)
    print(io, "\n  native: ")
    if valid_moments(D, Val(names), d.vals)::Bool
        show(io, to_native(D, Val(names), d.vals))
    else
        print(io, "invalid parameters")
    end
    return nothing
end
