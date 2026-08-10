# Closed-form conversions from a family's alternative parameters to its
# native ones, built with `check_args = false`. `valid_moments` is checked
# at every call site before `to_native` runs, so an invalid point yields
# `-Inf` rather than an error raised mid-gradient.

# LogNormal by the mean and standard deviation of the distribution itself,
# rather than of its logarithm. Inverting the log-normal moments,
#   mean = exp(mu + sigma^2 / 2),  var = mean^2 * (exp(sigma^2) - 1)
# gives sigma^2 = log1p((sd / mean)^2) and mu = log(mean) - sigma^2 / 2.
# `log1p` keeps the small-`sd / mean` case accurate.
function to_native(::Type{LogNormal}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    s2 = log1p((sd / mean)^2)
    return LogNormal(log(mean) - s2 / 2, sqrt(s2); check_args = false)
end

# The conversion squares `sd / mean`, so a negative `sd` would otherwise
# build exactly the same, valid native distribution as a positive one.
function valid_moments(::Type{LogNormal}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    return mean > 0 && sd > 0
end

# The same, given the variance instead of the standard deviation.
function to_native(::Type{LogNormal}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return to_native(LogNormal, Val((:mean, :sd)), (mean, sqrt(var)))
end

# Every `(mean, var)` predicate checks `var > 0` then delegates to the
# `(mean, sd)` predicate at `sqrt(var)` (short-circuited, so `sqrt` never
# sees a negative input), rather than restating the condition in variance
# coordinates, which can disagree with `sqrt(var)^2` in the last bit.
function valid_moments(::Type{LogNormal}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return var > 0 && valid_moments(LogNormal, Val((:mean, :sd)),
        (mean, sqrt(var)))
end

# Gamma by mean and standard deviation. A Gamma(shape, scale) has
#   mean = shape * scale,  var = shape * scale^2
# so scale = var / mean and shape = mean / scale = mean^2 / var.
function to_native(::Type{Gamma}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    scale = sd^2 / mean
    return Gamma(mean / scale, scale; check_args = false)
end

function valid_moments(::Type{Gamma}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    return mean > 0 && sd > 0
end

function to_native(::Type{Gamma}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return to_native(Gamma, Val((:mean, :sd)), (mean, sqrt(var)))
end

function valid_moments(::Type{Gamma}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return var > 0 && valid_moments(Gamma, Val((:mean, :sd)),
        (mean, sqrt(var)))
end

# Gamma by mean and shape: the shape is native, so only the scale is
# derived, `scale = mean / shape`.
function to_native(::Type{Gamma}, ::Val{(:mean, :shape)}, vals)
    mean, shape = vals
    return Gamma(shape, mean / shape; check_args = false)
end

function valid_moments(::Type{Gamma}, ::Val{(:mean, :shape)}, vals)
    mean, shape = vals
    return mean > 0 && shape > 0
end

# NegativeBinomial by mean and overdispersion `a`, the excess variance
# relative to a Poisson: var = mean + a * mean^2. The native
# `NegativeBinomial(r, p)` has mean = r(1-p)/p, var = mean/p, giving
#   r = 1 / a,  p = mean / var = 1 / (1 + a * mean).
#
# The family is DISCRETE; the wrapper takes its value support from it, so
# this stays discrete rather than silently becoming continuous.
function to_native(::Type{NegativeBinomial},
        ::Val{(:mean, :overdispersion)}, vals)
    mean, a = vals
    p = 1 / (1 + a * mean)
    return NegativeBinomial(1 / a, p; check_args = false)
end

# `a = 0` is the Poisson limit, not a NegativeBinomial: `r = 1 / a` diverges,
# so it is rejected alongside `a < 0`.
function valid_moments(::Type{NegativeBinomial},
        ::Val{(:mean, :overdispersion)}, vals)
    mean, a = vals
    return mean > 0 && a > 0
end

# NegativeBinomial by mean and dispersion, the reciprocal of overdispersion:
#   var = mean + mean^2 / dispersion
# The native `NegativeBinomial(r, p)` is reached directly: `r` is the
# dispersion and `p = r / (r + mean)`.
#
# The canonical (sorted) name order is `(:dispersion, :mean)`, not
# `(:mean, :dispersion)`: `_canonical` sorts alphabetically, and 'd' < 'm'.
function to_native(::Type{NegativeBinomial},
        ::Val{(:dispersion, :mean)}, vals)
    dispersion, mean = vals
    p = dispersion / (dispersion + mean)
    return NegativeBinomial(dispersion, p; check_args = false)
end

function valid_moments(::Type{NegativeBinomial},
        ::Val{(:dispersion, :mean)}, vals)
    dispersion, mean = vals
    return dispersion > 0 && mean > 0
end

# Exponential by rate: the native `Exponential(θ)` takes the scale, so
# θ = 1 / rate.
function to_native(::Type{Exponential}, ::Val{(:rate,)}, vals)
    rate, = vals
    return Exponential(1 / rate; check_args = false)
end

function valid_moments(::Type{Exponential}, ::Val{(:rate,)}, vals)
    rate, = vals
    return rate > 0
end

# Gamma by shape and rate: the shape is native, and the rate inverts to a
# scale, `scale = 1 / rate`.
#
# The canonical (sorted) name order is `(:rate, :shape)`, not
# `(:shape, :rate)`: `_canonical` sorts alphabetically, and 'r' < 's'.
function to_native(::Type{Gamma}, ::Val{(:rate, :shape)}, vals)
    rate, shape = vals
    return Gamma(shape, 1 / rate; check_args = false)
end

function valid_moments(::Type{Gamma}, ::Val{(:rate, :shape)}, vals)
    rate, shape = vals
    return rate > 0 && shape > 0
end

# SkewNormal by centre, scale and the probability mass below the centre, an
# elicitation form with an exact closed-form inversion.
#
# The native `SkewNormal(xi, omega, alpha)` has location `xi`, scale `omega`
# and shape `alpha`. For the untruncated family,
#   P(X < xi) = 1/2 - atan(alpha) / pi
# which inverts to
#   alpha = tan(pi * (1/2 - mass_below_centre)),  0 < mass_below_centre < 1.
#
# The canonical (sorted) name order is `(:centre, :mass_below_centre, :scale)`.
function to_native(::Type{SkewNormal},
        ::Val{(:centre, :mass_below_centre, :scale)}, vals)
    centre, m, scale = vals
    alpha = tan(pi * (1 / 2 - m))
    return SkewNormal(centre, scale, alpha; check_args = false)
end

function valid_moments(::Type{SkewNormal},
        ::Val{(:centre, :mass_below_centre, :scale)}, vals)
    centre, m, scale = vals
    return scale > 0 && 0 < m < 1
end

# Beta by mean and standard deviation. A Beta(alpha, beta) has
#   mean = alpha / (alpha + beta),  var = mean * (1 - mean) / (nu + 1)
# writing nu = alpha + beta. So
#   nu = mean * (1 - mean) / var - 1,
#   alpha = mean * nu,  beta = (1 - mean) * nu.
function to_native(::Type{Beta}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    nu = mean * (1 - mean) / sd^2 - 1
    return Beta(mean * nu, (1 - mean) * nu; check_args = false)
end

# `nu > 0` is exactly `var < mean * (1 - mean)`: the variance of any Beta is
# bounded above by that of a Bernoulli with the same mean, so a standard
# deviation elicited too wide for its mean has no Beta at all.
function valid_moments(::Type{Beta}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    return 0 < mean < 1 && sd > 0 && sd^2 < mean * (1 - mean)
end

function to_native(::Type{Beta}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return to_native(Beta, Val((:mean, :sd)), (mean, sqrt(var)))
end

function valid_moments(::Type{Beta}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return var > 0 && valid_moments(Beta, Val((:mean, :sd)),
        (mean, sqrt(var)))
end

# InverseGaussian by mean and standard deviation. The native
# `InverseGaussian(mu, lambda)` already takes the mean as `mu`, so only the
# shape needs inverting: mean = mu, var = mu^3 / lambda gives
#   lambda = mean^3 / var.
function to_native(::Type{InverseGaussian}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    lambda = mean^3 / sd^2
    return InverseGaussian(mean, lambda; check_args = false)
end

function valid_moments(::Type{InverseGaussian}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    return mean > 0 && sd > 0
end

function to_native(::Type{InverseGaussian}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return to_native(InverseGaussian, Val((:mean, :sd)), (mean, sqrt(var)))
end

function valid_moments(::Type{InverseGaussian}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return var > 0 && valid_moments(InverseGaussian, Val((:mean, :sd)),
        (mean, sqrt(var)))
end

# Weibull by mean and standard deviation. Unlike every family above, this has
# no exact closed form: a `Weibull(a, b)` (shape `a`, scale `b`) has raw
# moments `E[X^n] = b^n * Γ(1 + n/a)`, so
#
#   mean = b * Γ(1 + 1/a)
#   cv^2 = Γ(1 + 2/a) / Γ(1 + 1/a)^2 - 1
#
# The scale cancels out of the CV relation, so the shape is pinned by one
# bounded, monotone equation in the CV alone, solved via `solve_moment`
# (numeric.jl); the scale then follows in closed form.
#
# Solved in `s = log(a)` and `u = exp(-s) = 1/a`, taking logs of the CV
# relation to avoid `Γ(1 + 2/a)` overflowing at small shapes:
#
#   F(s) = logΓ(1 + 2u) - 2 logΓ(1 + u) - log1p(cv^2)
#
# `dF/ds = 2u * [digamma(1+u) - digamma(1+2u)] < 0` everywhere, since
# `digamma` is increasing and `1 + 2u > 1 + u` for `u > 0`: the CV is a
# strictly decreasing, bijective function of the shape, so any
# sign-changing bracket finds the unique root safely.
function _weibull_residual(s, vals)
    mean, sd = vals
    u = exp(-s)
    return loggamma(1 + 2u) - 2 * loggamma(1 + u) - log1p((sd / mean)^2)
end

function _weibull_deriv(s, vals)
    u = exp(-s)
    return 2u * (digamma(1 + u) - digamma(1 + 2u))
end

# The bracket is built in the promoted input type, so a `Float32` wrapper
# solves, and comes back, in `Float32` rather than silently widening.
#
# `shape_min` and `shape_max` are measured, not derived: both the scale
# completion in `to_native` and Distributions.jl's own `var(::Weibull)`
# overflow before the residual equation itself does, and near `shape_max`
# a residual within tolerance can still miss the requested CV by more than
# that tolerance implies.
_weibull_shape_min(::Type{T}) where {T} = T(0.0125)
_weibull_shape_max(::Type{T}) where {T} = T(5_000.0)

function _weibull_bracket(pvals)
    T = eltype(pvals)
    return log(_weibull_shape_min(T)), log(_weibull_shape_max(T))
end

# Both CV bounds are the CV attained at the bracket's own ends, so they are
# derived from the bracket rather than hand-tuned.
function _weibull_cv_min(::Type{T}) where {T}
    return T(pi) / (sqrt(T(6)) * _weibull_shape_max(T))
end

# Computed via `exp(x / 2)` rather than `sqrt(expm1(x))`: `x` (~136 in
# Float64) makes `exp(x)` overflow a narrow type before `sqrt` could bring
# it back down.
function _weibull_cv_max(::Type{T}) where {T}
    a = _weibull_shape_min(T)
    x = loggamma(1 + 2 / a) - 2 * loggamma(1 + 1 / a)
    return exp(x / 2)
end

# The window the root-find can actually solve, checked before `to_native`
# runs the solve, so an out-of-window request is `-Inf` rather than
# reaching `_check_bracket`'s throw.
function valid_moments(::Type{Weibull}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    (mean > 0 && sd > 0) || return false
    cv = sd / mean
    T = typeof(cv)
    return _weibull_cv_min(T) < cv < _weibull_cv_max(T)
end

# The shape follows from the solved root; the scale then follows in closed
# form, `b = mean / Γ(1 + 1/a) = mean * exp(-logΓ(1 + u))`.
function to_native(::Type{Weibull}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    s = solve_moment(Weibull, Val((:mean, :sd)), _weibull_residual,
        _weibull_deriv, _weibull_bracket, vals)
    u = exp(-s)
    return Weibull(exp(s), mean * exp(-loggamma(1 + u)); check_args = false)
end

function to_native(::Type{Weibull}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return to_native(Weibull, Val((:mean, :sd)), (mean, sqrt(var)))
end

function valid_moments(::Type{Weibull}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return var > 0 && valid_moments(Weibull, Val((:mean, :sd)),
        (mean, sqrt(var)))
end
