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
function _constraint_arity(::Type)
    return nothing
end
function _constraint_arity(::Type{<:_LocationScale})
    return 2
end
function _constraint_arity(::Type{Exponential})
    return 1
end

# The moment names that inversion can express, alongside any probabilities.
# An Exponential's mean is not among them: it is the family's only
# parameter, so `(:mean,)` is a whole constraint set and is registered
# concretely below rather than reached as a row. Its standard deviation is
# that same scale, and is reached as a row like any other.
function _moment_names(::Type)
    return ()
end
function _moment_names(::Type{<:_LocationScale})
    return (:median,)
end
function _moment_names(::Type{<:_LinearMoments})
    return (:mean, :median, :sd)
end
function _moment_names(::Type{Exponential})
    return (:median, :sd)
end

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
function _std_sd(::Type{Normal})
    return 1.0
end
function _std_sd(::Type{Logistic})
    return pi / sqrt(3)
end
function _std_sd(::Type{Laplace})
    return sqrt(2)
end
function _std_sd(::Type{Uniform})
    return 1 / sqrt(12)
end

# The transform taking the variate into the location-scale space.
_ls_transform(::Type, x) = x
_ls_transform(::Type{LogNormal}, x) = log(x)

# Constraint values the transform is undefined at, screened before the solve
# rather than after: `log` throws on a non-positive argument, and
# `valid_moments` must answer without throwing.
function _in_domain(::Type, v)
    return true
end
function _in_domain(::Type{LogNormal}, v)
    return v > 0
end

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

function _constraint_row(::Type{D}, ::Val{:sd}, v) where {D <: _LinearMoments}
    return (zero(v), oftype(v, _std_sd(D)), v)
end

# An `Exponential(theta)` has `sd = theta`, so its standard deviation is its
# scale outright.
function _constraint_row(::Type{Exponential}, ::Val{:sd}, v)
    return (zero(v), one(v), v)
end

# Every constraint's row. Generated for the reason `_canonical` is: a moment
# name becomes a `Val` and a probability a literal at compile time, so
# nothing indexes the name tuple or compares a `Symbol` on the sampler's hot
# path.
#
# The generator reads `names` and nothing else. It does not consult the
# method table — the `_constraint_row` calls it emits are resolved by
# ordinary dispatch afterwards — so a method added later is picked up
# normally. That is the property `_convertible` below could not have, which
# is why it is not generated.
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
# inversion expresses, every probability is a distinct point strictly inside
# (0, 1), and the constraints are independent of each other.
#
# Deliberately NOT a generated function, though everything it reads is known
# at compile time. A generator's result is cached per signature and is not
# invalidated when a method is added afterwards, so a generated body that
# called `_constraint_arity` or `_moment_names` would answer from whichever
# methods existed the first time it ran and keep that answer for the rest of
# the session. Written this way it is ordinary dispatch, and folds to a
# constant just the same: `names` is a type parameter, so the recursion
# below unrolls and the whole call collapses at compile time.
function _convertible(::Type{D}, ::Val{names}) where {D, names}
    arity = _constraint_arity(D)
    arity === nothing && return false
    length(names) == arity || return false
    _check_names(D, names) || return false
    return _independent(D, Val(names))
end

# Whether the constraints say independent things. Each row's COEFFICIENTS
# depend only on the names, never on the values, so this is a structural
# fact about the request: a mean and a median of a family symmetric about
# its location are one row written twice, and no numbers put into them can
# make the pair describe a distribution.
function _independent(::Type{D}, ::Val{names}) where {D, names}
    return _full_rank(_rows(D, Val(names), ntuple(_ -> 1.0, length(names))))
end

_full_rank(rows::Tuple{Any}) = !iszero(rows[1][2])

function _full_rank(rows::Tuple{Any, Any})
    (a1, b1, _), (a2, b2, _) = rows
    return !iszero(a1 * b2 - a2 * b1)
end

# Recursion over the tuple rather than an index loop: each step is
# concretely typed, so nothing indexes a heterogeneous tuple at run time.
function _check_names(::Type{D}, names::Tuple) where {D}
    rest = Base.tail(names)
    _check_name(D, first(names), rest) || return false
    return isempty(rest) || _check_names(D, rest)
end

function _check_name(::Type{D}, n::Symbol, rest) where {D}
    return n in _moment_names(D)
end

# A probability has to be a point strictly inside (0, 1), and has to appear
# once: two constraints at one probability say the same thing twice.
function _check_name(::Type{D}, p::Real, rest) where {D}
    return 0 < p < 1 && !any(q -> q isa Real && q == p, rest)
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
        n = length(names)
        return "$(D) is pinned by $(arity) constraint$(arity == 1 ? "" : "s")" *
               ", but $(n) $(n == 1 ? "was" : "were") given: ($(spec))"
    end
    probs = filter(!(n -> n isa Symbol), collect(names))
    if !all(p -> 0 < p < 1, probs)
        return "a quantile probability must lie strictly inside (0, 1); " *
               "got $(probs)"
    elseif !allunique(probs)
        return "a quantile probability must appear once; got $(probs)"
    elseif _check_names(D, names)
        return "($(spec)) are not independent constraints on $(D): they " *
               "pin the same combination of its parameters, so no values " *
               "of them describe a member of the family"
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

# The DERIVED native parameters, not merely the constraint values: two
# quantiles far enough apart overflow the scale, and two close enough
# together collapse it, on constraints that each looked perfectly ordinary.
# The #93/#94 guard, in constraint coordinates.
_native_ok(::Type, mu, sigma) = isfinite(mu) && isfinite(sigma) && sigma > 0

# `Uniform(a, b)` carries the width in its second native parameter, so the
# quantity that has to survive is `mu + sigma`. `mu < b` is also what
# catches a width too small to register against its own location, where
# `sigma > 0` alone still passes.
function _native_ok(::Type{Uniform}, mu, sigma)
    b = mu + sigma
    return isfinite(mu) && isfinite(b) && mu < b
end

function _location_scale_valid(::Type{D}, ::Val{names},
        vals) where {D, names}
    all(v -> _in_domain(D, v), vals) || return false
    mu, sigma = _location_scale(D, Val(names), vals)
    return _native_ok(D, mu, sigma)
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
    return _location_scale_valid(D, Val(names), vals)
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

# --- The standard moments, registered concretely ----------------------------
#
# The methods above fix the FAMILY and leave the names free. The
# standard-moment fallback is the mirror image: it fixes the names —
# `(:mean, :sd)`, `(:mean, :var)`, `(:mean,)` — and leaves the family free.
# It arrives with #106, as `src/standard_moments.jl`, so that path is a
# FORWARD reference and is not in this tree yet. Neither is more specific
# than the other, so every (one of these families, one of those name
# tuples) pair is ambiguous unless something more specific than both is
# registered for it.
#
# That is what the rest of this section is. Each method fixes both, so it
# beats the pair by ordinary specificity, and each is the conversion the
# family deserves anyway: exact algebra where the moments pin the family,
# and a refusal naming the reason where they cannot.

# Registered over the WHOLE family union rather than over the families
# whose moments happen to be linear, with the exceptions branched on
# inside. Julia reports an ambiguity unless a single method covers the
# entire intersection: coverage split across one method for most families
# and another for the rest does not count, even though every concrete call
# dispatches unambiguously.
#
# `D` is a static parameter, so the branch is settled at compile time and
# the return type stays concrete family by family.
#
# Only `_LinearMoments` and `Cauchy` need an answer: LogNormal is the rest
# of the union, and its own `(:mean, :sd)` and `(:mean, :var)` methods in
# families.jl are more specific than anything here, so it never arrives.
_assert_standard_moments(::Type{<:_LinearMoments}) = nothing
_assert_standard_moments(::Type{Cauchy}) = _no_cauchy_moments()

# A location and a scale are what these families' mean and standard
# deviation already are, so the shared constraint solve answers this
# exactly and no root-find is needed.
function to_native(::Type{D}, ::Val{(:mean, :sd)},
        vals) where {D <: _LocationScale}
    _assert_standard_moments(D)
    mu, sigma = _location_scale(D, Val((:mean, :sd)), vals)
    return _from_location_scale(D, mu, sigma)
end

function valid_moments(::Type{D}, ::Val{(:mean, :sd)},
        vals) where {D <: _LocationScale}
    D <: _LinearMoments || return true
    return _location_scale_valid(D, Val((:mean, :sd)), vals)
end

function to_native(::Type{D}, ::Val{(:mean, :var)},
        vals) where {D <: _LocationScale}
    mean, var = vals
    return to_native(D, Val((:mean, :sd)), (mean, sqrt(var)))
end

function valid_moments(::Type{D}, ::Val{(:mean, :var)},
        vals) where {D <: _LocationScale}
    D <: _LinearMoments || return true
    mean, var = vals
    return var > 0 && valid_moments(D, Val((:mean, :sd)), (mean, sqrt(var)))
end

# An `Exponential(theta)` has `mean = theta`, so the mean is the scale.
function to_native(::Type{Exponential}, ::Val{(:mean,)}, vals)
    return Exponential(vals[1]; check_args = false)
end

function valid_moments(::Type{Exponential}, ::Val{(:mean,)}, vals)
    mean, = vals
    return isfinite(mean) && mean > 0
end

# The refusals. Each raises rather than converting, so its inferred return
# type is `Union{}` — see the registered-pair sweep in test/families.jl,
# which holds a registered pair to converting concretely OR refusing
# outright, and lists which are which.

# One constraint leaves a two-parameter family a one-parameter family
# short, whatever that constraint is.
function to_native(::Type{D}, ::Val{(:mean,)}, vals) where {D <: _LocationScale}
    throw(ArgumentError(_unconvertible_message(D, (:mean,))))
end

function valid_moments(::Type{D}, ::Val{(:mean,)},
        vals) where {D <: _LocationScale}
    return true
end

# A Cauchy's mean and variance are both undefined, so no Cauchy has the
# ones asked for and no solve can find one.
function _no_cauchy_moments()
    throw(ArgumentError(
        "a Cauchy has neither a mean nor a standard deviation, so no Cauchy " *
        "is pinned by them; elicit it by quantiles, or by a median and a " *
        "quantile, instead"))
end

# An `Exponential(theta)` has `sd = mean = theta`, so a mean and a standard
# deviation are one constraint too many and agree only by coincidence.
function _exponential_over_determined()
    throw(ArgumentError(
        "an Exponential is pinned by one parameter, and has a standard " *
        "deviation equal to its mean, so a mean and a standard deviation " *
        "over-determine it; give the mean alone, or the rate"))
end

function to_native(::Type{Exponential}, ::Val{(:mean, :sd)}, vals)
    return _exponential_over_determined()
end

function to_native(::Type{Exponential}, ::Val{(:mean, :var)}, vals)
    return _exponential_over_determined()
end

function valid_moments(::Type{Exponential}, ::Val{(:mean, :sd)}, vals)
    return true
end
function valid_moments(::Type{Exponential}, ::Val{(:mean, :var)}, vals)
    return true
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
