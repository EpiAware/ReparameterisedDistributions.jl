# The standard-moment fallback: `(:mean, :sd)`, `(:mean, :var)` and
# `(:mean,)` are answered for ANY family, by solving the family's own
# moment equations for its native parameters whenever no closed form is
# registered for that pair. Nothing is registered per family and no trait
# is opted into; a family's own conversion is strictly more specific in its
# first argument, so every exact algebra in families.jl still wins by
# ordinary dispatch.
#
# The solve runs in unconstrained coordinates, one per native parameter,
# under the transform `native_domains` reports. That same tuple's length is
# the family's native parameter count, so a family that overrides it fixes
# the transform and the arity together.

@doc raw"

The domain of each of a family's native parameters, and — through the
tuple's length — how many native parameters it has.

Read only by the standard-moment fallback, which solves for the native
parameters of a family that registers no closed form. Each entry is one of

- `:positive`, solved as `exp(z)`,
- `:real`, solved as `z`,
- `:unit`, solved as `logistic(z)`,

so the iteration cannot leave the family's own parameter space, and the
length says how many moments pin the family down: supplying any other
number raises rather than fitting some of them.

The default reports every native parameter as `:positive`, and takes the
count from the family's own fields. That covers a shape-and-scale family
(`Frechet`, `Pareto`, `InverseGamma`, …) as it stands. A family with a
location, a probability, or a parameter count its fields do not give
registers a method here — one line, and the fallback then works for it.

# Arguments
- the native family being converted to.

# Returns
A tuple of `Symbol`s, one per native parameter, in the order the family's
own constructor takes them.

# Examples
```@example
using ReparameterisedDistributions, Distributions
using ReparameterisedDistributions: native_domains

native_domains(Frechet), native_domains(Normal)
```

A family whose location is unconstrained registers it directly:

```julia
ReparameterisedDistributions.native_domains(::Type{Laplace}) =
    (:real, :positive)
```

# See also
- [`to_native`](@ref): the conversion the fallback supplies for a family
  that registers none.
- [`solve_moments`](@ref): the driver the fallback runs.
"
function native_domains(::Type{D}) where {D}
    return ntuple(_ -> :positive, Val(fieldcount(Base.unwrap_unionall(D))))
end

# The unconstrained coordinate and back. Branching on a `Symbol` rather
# than dispatching on a singleton keeps this type-stable whether or not
# the domains constant-fold: every branch returns the same type.
function _unlink(domain::Symbol, z)
    domain === :real && return z
    domain === :unit && return inv(one(z) + exp(-z))
    return exp(z)
end

function _link(domain::Symbol, θ)
    domain === :real && return θ
    domain === :unit && return log(θ / (one(θ) - θ))
    return log(θ)
end

function _native_values(domains::NTuple{N, Symbol}, z::NTuple{N}) where {N}
    return ntuple(i -> _unlink(domains[i], z[i]), Val(N))
end

# A moment that is undefined at the current iterate (a negative variance
# from a parameter the transform admits but the family does not) comes
# back as NaN, which the line search rejects, rather than as a raised
# `DomainError` from `log`.
_poslog(x) = x > 0 ? log(x) : oftype(float(x), NaN)

# One moment per native parameter, decided from the names and the family
# alone, so the branch folds away and the check costs nothing per call.
function _check_moment_arity(::Type{D}, ::Val{names}) where {D, names}
    n = length(native_domains(D))
    length(names) == n && return nothing
    throw(ArgumentError(
        "$(D) has $(n) native parameters, and $(length(names)) standard " *
        "moments $(collect(names)) do not pin them down; no closed form " *
        "is registered for this pair, and the numeric fallback needs one " *
        "moment per native parameter"))
end

# The moment equations, scaled so both are relative: the mean by the
# larger of the requested mean and standard deviation (which keeps a
# requested mean of zero well defined), the standard deviation in logs.
# An unscaled pair lets one equation dominate the line search whenever the
# coefficient of variation is far from one.
function _standard_residual(::Type{D}, ::Val{(:mean, :sd)}, domains, z,
        vals) where {D}
    m, sd = vals
    nd = D(_native_values(domains, z)...; check_args = false)
    return ((mean(nd) - m) / max(abs(m), sd),
        (_poslog(var(nd)) - 2 * log(sd)) / 2)
end

function _standard_residual(::Type{D}, ::Val{(:mean,)}, domains, z,
        vals) where {D}
    m, = vals
    nd = D(_native_values(domains, z)...; check_args = false)
    return ((mean(nd) - m) / max(abs(m), one(m)),)
end

# Starting points, tried in order. The first is the one that is exact for
# a location-scale family and close for a shape-and-scale one: a location
# takes the requested mean, a probability a half, and a scale-like
# positive parameter (the last of them) the requested magnitude, leaving
# any shape at one. The other two are the flat alternatives, all ones and
# all at the requested magnitude, for a family the first misses.
function _seed(domains::NTuple{N, Symbol}, m, positive::F) where {N, F}
    return ntuple(Val(N)) do i
        domain = domains[i]
        θ = domain === :real ? m :
            domain === :unit ? oftype(m, 0.5) : positive(i)
        _link(domain, θ)
    end
end

function _seed_ladder(domains::NTuple{N, Symbol}, m, sd) where {N}
    scale = max(abs(m), sd)
    located = any(==(:real), domains)
    return (_seed(domains, m, i -> located ? sd : (i == N ? scale : one(m))),
        _seed(domains, m, i -> one(m)),
        _seed(domains, m, i -> scale))
end

function _standard_seeds(::Val{(:mean, :sd)}, domains, vals)
    m, sd = vals
    return _seed_ladder(domains, m, sd)
end

function _standard_seeds(::Val{(:mean,)}, domains, vals)
    m, = vals
    return _seed_ladder(domains, m, max(abs(m), one(m)))
end

# Whether the iteration reaches the moments asked for. Answering this
# honestly means running the same solve the conversion runs, so a request
# outside the solvable region costs the iteration twice: once here, once
# in `to_native`. The alternative — reporting `true` unconditionally and
# letting the solver throw — would break the `-Inf` contract that lets a
# sampler probe an invalid point without raising mid-gradient.
function _moments_solvable(::Type{D}, v::Val{names}, vals) where {D, names}
    domains = native_domains(D)
    pvals = map(_primal, vals)
    _, _, converged = _solve_moment_system(
        z -> _standard_residual(D, v, domains, z, pvals),
        _standard_seeds(v, domains, pvals))
    return converged
end

function _solve_standard(::Type{D}, v::Val{names}, vals) where {D, names}
    domains = native_domains(D)
    z = solve_moments(D, v,
        (zz, w) -> _standard_residual(D, v, domains, zz, w),
        pvals -> _standard_seeds(v, domains, pvals), vals)
    return _native_values(domains, z)
end

function to_native(::Type{D}, v::Val{(:mean, :sd)}, vals) where {D}
    _check_moment_arity(D, v)
    return D(_solve_standard(D, v, vals)...; check_args = false)
end

function valid_moments(::Type{D}, v::Val{(:mean, :sd)}, vals) where {D}
    _check_moment_arity(D, v)
    m, sd = vals
    (isfinite(m) && sd > 0 && isfinite(sd)) || return false
    return _moments_solvable(D, v, vals)
end

function to_native(::Type{D}, v::Val{(:mean,)}, vals) where {D}
    _check_moment_arity(D, v)
    return D(_solve_standard(D, v, vals)...; check_args = false)
end

function valid_moments(::Type{D}, v::Val{(:mean,)}, vals) where {D}
    _check_moment_arity(D, v)
    m, = vals
    isfinite(m) || return false
    return _moments_solvable(D, v, vals)
end

# `(:mean, :var)` delegates at `sqrt(var)`, as every registered family
# does, so a family reached here and a family with a closed form for
# `(:mean, :sd)` are both answered in variance coordinates by the same
# route.
function to_native(::Type{D}, ::Val{(:mean, :var)}, vals) where {D}
    m, v = vals
    return to_native(D, Val((:mean, :sd)), (m, sqrt(v)))
end

function valid_moments(::Type{D}, ::Val{(:mean, :var)}, vals) where {D}
    m, v = vals
    return v > 0 && valid_moments(D, Val((:mean, :sd)), (m, sqrt(v)))
end
