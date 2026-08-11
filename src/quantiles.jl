# Quantile values as parameters: a family whose parameters are the values of
# elicited quantiles, on their own or mixed with moment names.
#
# A probability sits in the `names` tuple where a `Symbol` would, so a mixed
# constraint set is an ordinary `Reparameterised` and reaches `to_native`
# through the same dispatch as a moment set. `_canonical` puts moment names
# before probabilities and orders probabilities ascending.

# --- The families with an exact inversion -----------------------------------

# A location-scale family: `Q(p) = t⁻¹(mu + sigma * z(p))` for a standard
# quantile `z` and a monotone transform `t`, so any two constraints linear
# in `(mu, sigma)` pin it exactly.
const _LocationScale = Union{Cauchy, Laplace, LogNormal, Logistic, Normal,
    Uniform}

# Those of the above whose mean and standard deviation are themselves linear
# in `(mu, sigma)`, so either can stand in a constraint set. Cauchy has
# neither; LogNormal has both, but `mean = exp(mu + sigma^2 / 2)` is linear
# in neither.
const _LinearMoments = Union{Laplace, Logistic, Normal, Uniform}

# The number of constraints a family's exact inversion needs, which is its
# native parameter count, or `nothing` where no exact inversion is
# registered. `nothing` is the seam the generic numeric solve plugs into.
_constraint_arity(::Type) = nothing
_constraint_arity(::Type{<:_LocationScale}) = 2
_constraint_arity(::Type{Exponential}) = 1

# The moment names that inversion can express, alongside any probabilities.
_moment_names(::Type) = ()
_moment_names(::Type{<:_LocationScale}) = (:median,)
_moment_names(::Type{<:_LinearMoments}) = (:mean, :median, :sd)
_moment_names(::Type{Exponential}) = (:mean, :median)

# --- The standard member ----------------------------------------------------

# The standard member's quantile in the transformed space. The probability
# is fixed by the parameterisation rather than estimated, so this always
# runs in plain floating point and never carries a derivative.
_std_quantile(::Type{<:Union{LogNormal, Normal}}, p) = quantile(Normal(), p)
_std_quantile(::Type{Logistic}, p) = log(p / (1 - p))
_std_quantile(::Type{Cauchy}, p) = tan(pi * (p - 1 / 2))
_std_quantile(::Type{Laplace}, p) = p < 1 / 2 ? log(2p) : -log(2 * (1 - p))
_std_quantile(::Type{Uniform}, p) = p
_std_quantile(::Type{Exponential}, p) = -log1p(-p)

# The standard member's own standard deviation, so an `sd` constraint is a
# row in the scale.
_std_sd(::Type{Normal}) = 1.0
_std_sd(::Type{Logistic}) = pi / sqrt(3)
_std_sd(::Type{Laplace}) = sqrt(2)
_std_sd(::Type{Uniform}) = 1 / sqrt(12)

# The transform taking the variate into the location-scale space.
_ls_transform(::Type, x) = x
_ls_transform(::Type{LogNormal}, x) = log(x)

# Constraint values the transform is undefined at, screened before the solve
# rather than after: `log` throws on a non-positive argument, and
# `valid_moments` must answer without throwing.
_in_domain(::Type, v) = true
_in_domain(::Type{LogNormal}, v) = v > 0

# Building the native distribution from the solved location and scale.
_from_location_scale(::Type{Normal}, mu, s) = Normal(mu, s; check_args = false)
function _from_location_scale(::Type{LogNormal}, mu, s)
    return LogNormal(mu, s; check_args = false)
end
function _from_location_scale(::Type{Logistic}, mu, s)
    return Logistic(mu, s; check_args = false)
end
function _from_location_scale(::Type{Cauchy}, mu, s)
    return Cauchy(mu, s; check_args = false)
end
function _from_location_scale(::Type{Laplace}, mu, s)
    return Laplace(mu, s; check_args = false)
end
# `z(p) = p` on a `Uniform(a, b)`, so the scale is the width.
function _from_location_scale(::Type{Uniform}, mu, s)
    return Uniform(mu, mu + s; check_args = false)
end

# --- Constraints as rows ----------------------------------------------------

# One constraint as a row `(a, b, r)` of `a * mu + b * sigma = r`.
function _constraint_row(::Type{D}, p::Real, v) where {D}
    return (one(v), oftype(v, _std_quantile(D, p)), _ls_transform(D, v))
end

# `Exponential(theta)` has no location, so its rows are in the scale alone.
function _constraint_row(::Type{Exponential}, p::Real, v)
    return (zero(v), oftype(v, _std_quantile(Exponential, p)), v)
end

# The median is the quantile at one half, whatever the family.
function _constraint_row(::Type{D}, ::Val{:median}, v) where {D}
    return _constraint_row(D, 1 / 2, v)
end

# The mean is the median in a family symmetric about its location.
function _constraint_row(::Type{D}, ::Val{:mean}, v) where {D <: _LinearMoments}
    return _constraint_row(D, 1 / 2, v)
end

_constraint_row(::Type{Exponential}, ::Val{:mean}, v) = (zero(v), one(v), v)

function _constraint_row(::Type{D}, ::Val{:sd}, v) where {D <: _LinearMoments}
    return (zero(v), oftype(v, _std_sd(D)), v)
end

# Every constraint's row. Generated for the reason `_canonical` is: a moment
# name becomes a `Val` and a probability a literal at compile time, so
# nothing indexes the name tuple or compares a `Symbol` on the sampler's hot
# path.
@generated function _rows(D::Type, ::Val{names}, vals) where {names}
    calls = [n isa Symbol ?
             :(_constraint_row(D, Val($(QuoteNode(n))), vals[$i])) :
             :(_constraint_row(D, $n, vals[$i]))
             for (i, n) in enumerate(names)]
    return Expr(:tuple, calls...)
end

# --- Which constraint sets invert exactly -----------------------------------

# Whether a constraint set is one this file can invert exactly: the family
# has an inversion, the count matches it, every moment name is one that
# inversion expresses, and every probability is a distinct point strictly
# inside (0, 1). Generated, so the answer is a literal `Bool`: it depends on
# nothing but the family and the names, and it is checked at every call site.
@generated function _convertible(D::Type, ::Val{names}) where {names}
    fam = D.parameters[1]
    fam isa Type || return :(false)
    arity = _constraint_arity(fam)
    arity == length(names) || return :(false)
    moments = _moment_names(fam)
    probs = filter(!(n -> n isa Symbol), collect(names))
    all(n -> n in moments, filter(n -> n isa Symbol, collect(names))) ||
        return :(false)
    all(p -> 0 < p < 1, probs) || return :(false)
    return :($(allunique(probs)))
end

# The seam the generic numeric solve for constraint sets plugs into. A set
# with no exact inversion — a Gamma elicited by two quantiles, a LogNormal
# by a mean and a tail point — reaches here, and the throw below is what the
# solve replaces: everything above it, the front door and the constraint
# rows alike, already carries such a set through unchanged.
function _assert_convertible(::Type{D}, ::Val{names}) where {D, names}
    _convertible(D, Val(names)) && return nothing
    throw(ArgumentError(_unconvertible_message(D, names)))
end

# Printed as it was typed: a moment as its name, a probability as the pair
# it came from.
_name_string(n::Symbol) = string(n)
_name_string(n::Real) = "quantile at $(n)"

function _unconvertible_message(::Type{D}, names) where {D}
    spec = join(map(_name_string, names), ", ")
    arity = _constraint_arity(D)
    if arity === nothing
        return "no reparameterisation of $(D) by ($(spec)) is registered: " *
               "$(D) has no exact inversion from quantile constraints, and " *
               "the generic numeric solve for constraint sets is not wired " *
               "up yet"
    elseif length(names) != arity
        return "$(D) is pinned by $(arity) constraint(s), but " *
               "$(length(names)) were given: ($(spec))"
    end
    probs = filter(!(n -> n isa Symbol), collect(names))
    if !all(p -> 0 < p < 1, probs)
        return "a quantile probability must lie strictly inside (0, 1); " *
               "got $(probs)"
    elseif !allunique(probs)
        return "a quantile probability must appear once; got $(probs)"
    end
    return "no reparameterisation of $(D) by ($(spec)) is registered: the " *
           "exact inversion of $(D) from constraints expresses " *
           "$(collect(_moment_names(D))) and quantile probabilities, and " *
           "the generic numeric solve for constraint sets is not wired up yet"
end

# --- The conversions --------------------------------------------------------

# Two rows solved by Cramer's rule. A singular system — two constraints
# saying the same thing — gives a non-finite scale, which `valid_moments`
# rejects before this runs.
function _location_scale(::Type{D}, ::Val{names}, vals) where {D, names}
    (a1, b1, r1), (a2, b2, r2) = _rows(D, Val(names), vals)
    det = a1 * b2 - a2 * b1
    return (r1 * b2 - r2 * b1) / det, (a1 * r2 - a2 * r1) / det
end

function to_native(::Type{D}, ::Val{names},
        vals) where {D <: _LocationScale, names}
    _assert_convertible(D, Val(names))
    mu, sigma = _location_scale(D, Val(names), vals)
    return _from_location_scale(D, mu, sigma)
end

# A set this file cannot invert stays `true` rather than `false`, so the
# conversion's own error surfaces instead of the request being silently
# zero-density. That matches the three-argument fallbacks, which pair an
# always-`true` predicate with an always-throwing conversion.
#
# A strictly increasing pair of values at strictly increasing probabilities
# is exactly a positive scale, so the ordering of an elicitation needs no
# check of its own.
function valid_moments(::Type{D}, ::Val{names},
        vals) where {D <: _LocationScale, names}
    _convertible(D, Val(names)) || return true
    all(v -> _in_domain(D, v), vals) || return false
    mu, sigma = _location_scale(D, Val(names), vals)
    return isfinite(mu) && isfinite(sigma) && sigma > 0
end

function to_native(::Type{Exponential}, ::Val{names}, vals) where {names}
    _assert_convertible(Exponential, Val(names))
    (_, b, r), = _rows(Exponential, Val(names), vals)
    return Exponential(r / b; check_args = false)
end

function valid_moments(::Type{Exponential}, ::Val{names}, vals) where {names}
    _convertible(Exponential, Val(names)) || return true
    (_, b, r), = _rows(Exponential, Val(names), vals)
    theta = r / b
    return isfinite(theta) && theta > 0
end

# --- The front door ---------------------------------------------------------

# The `quantiles` keyword, as `probability => value` pairs. A lone pair is
# the one-constraint case. A vector is refused: the number of constraints
# and the probabilities themselves belong in the wrapper's type, and a
# vector puts neither there.
_quantile_pairs(q::Pair{<:Real, <:Real}) = (q,)
_quantile_pairs(q::Tuple{Vararg{Pair{<:Real, <:Real}}}) = q
function _quantile_pairs(q)
    throw(ArgumentError(
        "`quantiles` takes `probability => value` pairs in a tuple, e.g. " *
        "quantiles = (0.05 => 1.2, 0.95 => 8.4), or a single pair; got " *
        "$(typeof(q))"))
end

# Splices the elicited quantiles into the same `(names, vals)` pair a moment
# keyword produces, then hands over to the shared construction path.
# `Float64` probabilities throughout, so two spellings of one elicitation do
# not build two wrapper types.
function _build_quantiles(::Type{D}, nt::NamedTuple;
        check_args::Bool = true) where {D}
    qs = _quantile_pairs(nt.quantiles)
    moments = Base.structdiff(nt, NamedTuple{(:quantiles,)})
    names = (keys(moments)..., map(q -> Float64(first(q)), qs)...)
    vals = (Tuple(moments)..., map(last, qs)...)
    _assert_convertible(D, Val(names))
    return _build(D, Val(names), vals; check_args = check_args)
end
