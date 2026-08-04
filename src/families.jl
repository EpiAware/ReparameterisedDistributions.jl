# Closed-form conversions from a family's alternative parameters to its native
# ones. Each is exact algebra, so it is differentiable and adds no solver. The
# native distribution is built with `check_args = false`; validity is decided in
# moment coordinates by `_valid_moments`, which the density consults so that an
# invalid point yields `-Inf` rather than an error raised mid-gradient.

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

# A log-normal is supported on the positives, so its mean is positive. The
# standard deviation must be checked in its own coordinates: the conversion
# squares `sd / mean`, so a negative one maps onto exactly the same valid native
# distribution as its positive counterpart.
function _valid_moments(::Type{LogNormal}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    return mean > 0 && sd > 0
end

# The same, given the variance instead of the standard deviation.
function to_native(::Type{LogNormal}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return to_native(LogNormal, Val((:mean, :sd)), (mean, sqrt(var)))
end

function _valid_moments(::Type{LogNormal}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return mean > 0 && var > 0
end

# Gamma by mean and standard deviation. A Gamma(shape, scale) has
#   mean = shape * scale,  var = shape * scale^2
# so scale = var / mean and shape = mean / scale = mean^2 / var.
function to_native(::Type{Gamma}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    scale = sd^2 / mean
    return Gamma(mean / scale, scale; check_args = false)
end

# As for the LogNormal, the conversion squares `sd`, so the sign has to be
# checked here or a negative standard deviation would give a valid — and
# identical — native distribution.
function _valid_moments(::Type{Gamma}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    return mean > 0 && sd > 0
end

function to_native(::Type{Gamma}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return to_native(Gamma, Val((:mean, :sd)), (mean, sqrt(var)))
end

function _valid_moments(::Type{Gamma}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return mean > 0 && var > 0
end

# Gamma by mean and shape, which is how a delay is often elicited when the shape
# carries the meaning (a fixed number of exponential stages, say). The shape is
# native, so only the scale is derived: scale = mean / shape. This is the pair
# CensoredDistributions registered.
function to_native(::Type{Gamma}, ::Val{(:mean, :shape)}, vals)
    mean, shape = vals
    return Gamma(shape, mean / shape; check_args = false)
end

function _valid_moments(::Type{Gamma}, ::Val{(:mean, :shape)}, vals)
    mean, shape = vals
    return mean > 0 && shape > 0
end

# NegativeBinomial by mean and overdispersion, the parameterisation epidemiology
# reaches for: the overdispersion `a` (a cluster factor) is the excess variance
# relative to a Poisson, through
#   var = mean + a * mean^2
# so a -> 0 recovers the Poisson limit and larger `a` means more clustering. The
# native `NegativeBinomial(r, p)` has mean = r(1-p)/p and var = mean/p, giving
#   r = 1 / a,  p = mean / var = 1 / (1 + a * mean).
#
# Note the family is DISCRETE. The wrapper takes its value support from the
# family, so this stays a discrete distribution rather than silently becoming a
# continuous one.
function to_native(::Type{NegativeBinomial},
        ::Val{(:mean, :overdispersion)}, vals)
    mean, a = vals
    p = 1 / (1 + a * mean)
    return NegativeBinomial(1 / a, p; check_args = false)
end

# `a = 0` is the Poisson limit, not a NegativeBinomial: `r = 1 / a` diverges.
function _valid_moments(::Type{NegativeBinomial},
        ::Val{(:mean, :overdispersion)}, vals)
    mean, a = vals
    return mean > 0 && a > 0
end

# NegativeBinomial by mean and dispersion, the reciprocal of the overdispersion
# above and at least as widely used a convention. Here
#   var = mean + mean^2 / dispersion
# so `dispersion -> Inf` recovers the Poisson limit rather than `-> 0`. The
# native `NegativeBinomial(r, p)` is reached directly: `r` is exactly the
# dispersion, and `p = r / (r + mean)`, which is the algebra a caller would
# otherwise have to invert from `overdispersion = 1 / dispersion` by hand —
# the confusion this pair exists to remove.
#
# The canonical (sorted) name order is `(:dispersion, :mean)`, not
# `(:mean, :dispersion)`: `_canonical` sorts alphabetically, and 'd' < 'm'.
function to_native(::Type{NegativeBinomial},
        ::Val{(:dispersion, :mean)}, vals)
    dispersion, mean = vals
    p = dispersion / (dispersion + mean)
    return NegativeBinomial(dispersion, p; check_args = false)
end

function _valid_moments(::Type{NegativeBinomial},
        ::Val{(:dispersion, :mean)}, vals)
    dispersion, mean = vals
    return dispersion > 0 && mean > 0
end

# Exponential by its rate, the quantity a hazard is usually written over and
# the quantity one wants reported, rather than the native scale (which is
# already the mean, but is still the reciprocal of the rate a prior is
# typically placed on). The native `Exponential(θ)` takes the scale directly,
# so the conversion is a single inversion: θ = 1 / rate.
function to_native(::Type{Exponential}, ::Val{(:rate,)}, vals)
    rate, = vals
    return Exponential(1 / rate; check_args = false)
end

function _valid_moments(::Type{Exponential}, ::Val{(:rate,)}, vals)
    rate, = vals
    return rate > 0
end

# Gamma by shape and rate, the reciprocal of the (mean, shape) pair above: the
# shape is native either way, and here the rate is native too once inverted to
# a scale, rather than the mean being. `scale = 1 / rate`.
#
# The canonical (sorted) name order is `(:rate, :shape)`, not
# `(:shape, :rate)`: `_canonical` sorts alphabetically, and 'r' < 's'.
function to_native(::Type{Gamma}, ::Val{(:rate, :shape)}, vals)
    rate, shape = vals
    return Gamma(shape, 1 / rate; check_args = false)
end

function _valid_moments(::Type{Gamma}, ::Val{(:rate, :shape)}, vals)
    rate, shape = vals
    return rate > 0 && shape > 0
end

# SkewNormal by its centre, scale and the probability mass below the centre —
# an elicitation form rather than a moment, but one with an exact closed-form
# inversion, so it keeps the package's contract of exact, differentiable,
# solver-free algebra.
#
# The native `SkewNormal(xi, omega, alpha)` has location `xi`, scale `omega`
# and shape `alpha`. For the UNTRUNCATED family the mass below the location
# depends only on the shape,
#   P(X < xi) = 1/2 - atan(alpha) / pi
# which inverts exactly to
#   alpha = tan(pi * (1/2 - mass_below_centre)),  0 < mass_below_centre < 1.
# This holds exactly only for the untruncated distribution; a caller who
# truncates the result gets an approximate, not exact, tail mass.
#
# Distributions.jl does not implement `cdf`/`quantile` for `SkewNormal`
# (Owen's T function is not implemented there), so that limitation is
# inherited by a wrapper built through this parameterisation exactly as it is
# by a native `SkewNormal`.
#
# The canonical (sorted) name order is `(:centre, :mass_below_centre, :scale)`.
function to_native(::Type{SkewNormal},
        ::Val{(:centre, :mass_below_centre, :scale)}, vals)
    centre, m, scale = vals
    alpha = tan(pi * (1 / 2 - m))
    return SkewNormal(centre, scale, alpha; check_args = false)
end

function _valid_moments(::Type{SkewNormal},
        ::Val{(:centre, :mass_below_centre, :scale)}, vals)
    centre, m, scale = vals
    return scale > 0 && 0 < m < 1
end

# Beta by mean and standard deviation, the natural coordinates for a
# probability-scale quantity elicited as a central value and an uncertainty
# (a reporting fraction or a case-fatality ratio, say) rather than as the
# native shape pair. A Beta(alpha, beta) has
#   mean = alpha / (alpha + beta),  var = mean * (1 - mean) / (nu + 1)
# writing nu = alpha + beta for the concentration. So
#   nu = mean * (1 - mean) / var - 1,
#   alpha = mean * nu,  beta = (1 - mean) * nu.
# `nu > 0` is exactly `var < mean * (1 - mean)`: the variance of any Beta is
# bounded above by the variance of a Bernoulli with the same mean, so a
# standard deviation elicited too wide for its mean has no Beta at all, not
# merely an invalid one, and is rejected rather than silently clipped.
function to_native(::Type{Beta}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    nu = mean * (1 - mean) / sd^2 - 1
    return Beta(mean * nu, (1 - mean) * nu; check_args = false)
end

function _valid_moments(::Type{Beta}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    return 0 < mean < 1 && sd > 0 && sd^2 < mean * (1 - mean)
end

function to_native(::Type{Beta}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return to_native(Beta, Val((:mean, :sd)), (mean, sqrt(var)))
end

function _valid_moments(::Type{Beta}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return 0 < mean < 1 && var > 0 && var < mean * (1 - mean)
end

# InverseGaussian by mean and standard deviation. The native
# `InverseGaussian(mu, lambda)` is already keyed on the mean (`mu`), so only
# the shape needs inverting: mean = mu and var = mu^3 / lambda give
#   lambda = mean^3 / var,
# an exact single-line inversion, unlike Gamma's, because the mean is native
# here already. The family is a first-passage-time distribution (hitting
# time of a drifting Wiener process), which makes it a genuine alternative to
# the Gamma and log-normal for a right-skewed delay such as an incubation
# period.
function to_native(::Type{InverseGaussian}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    lambda = mean^3 / sd^2
    return InverseGaussian(mean, lambda; check_args = false)
end

function _valid_moments(::Type{InverseGaussian}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    return mean > 0 && sd > 0
end

function to_native(::Type{InverseGaussian}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return to_native(InverseGaussian, Val((:mean, :sd)), (mean, sqrt(var)))
end

function _valid_moments(::Type{InverseGaussian}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return mean > 0 && var > 0
end

# Weibull by mean and standard deviation. Unlike every family above, this has
# no exact closed form: a `Weibull(a, b)` (shape `a`, scale `b`) has raw
# moments `E[X^n] = b^n * Γ(1 + n/a)`, so
#
#   mean = b * Γ(1 + 1/a)
#   cv^2 = Γ(1 + 2/a) / Γ(1 + 1/a)^2 - 1
#
# The scale cancels out of the CV relation, so the shape is pinned by one
# bounded, monotone, one-dimensional equation in the CV alone; once it is
# known the scale follows in closed form, `b = mean / Γ(1 + 1/a)`. This
# family opts into the numeric fallback (see numeric.jl) instead of adding a
# `to_native` method directly.
_conversion_kind(::Type{Weibull}, ::Val{(:mean, :sd)}) = Numeric()

# Solved in `s = log(a)` (scale-free, so the bracket is a plain interval and
# `a > 0` is automatic) and in `u = exp(-s) = 1/a`, taking logs of the CV
# relation to avoid `Γ(1 + 2/a)` overflowing at small shapes:
#
#   F(s) = logΓ(1 + 2u) - 2 logΓ(1 + u) - log1p(cv^2)
#
# `digamma` is strictly increasing on `(0, Inf)` and `1 + 2u > 1 + u` for
# `u > 0`, so `dF/ds = 2u * [digamma(1+u) - digamma(1+2u)] < 0` everywhere:
# the CV is a strictly decreasing, and hence bijective, function of the
# shape, so the root is unique and any sign-changing bracket is safe for a
# bracketing solver.
function _moment_residual(::Type{Weibull}, ::Val{(:mean, :sd)}, s, vals)
    mean, sd = vals
    u = exp(-s)
    return loggamma(1 + 2u) - 2 * loggamma(1 + u) - log1p((sd / mean)^2)
end

function _moment_residual_deriv(::Type{Weibull}, ::Val{(:mean, :sd)}, s,
        vals)
    u = exp(-s)
    return 2u * (digamma(1 + u) - digamma(1 + 2u))
end

# A fixed bracket in log-shape, built in the promoted input type so the
# solve runs in that type too (a `Float64` bracket under a `Dual`-valued
# residual would throw from inside the solver rather than silently
# truncate, but would also stop the bracket following the caller's number
# type as it must for a `Float32` wrapper to come back `Weibull{Float32}`).
# No expansion loop: A42 converges in roughly 10 evaluations regardless of
# how wide the bracket is.
#
# `shape_min` is NOT the loosest bound the residual equation alone could
# support (`log1p(cv^2)` only overflows once `cv` exceeds roughly
# `sqrt(typemax(T))`, around `1e154` in Float64). It is tighter, for two
# reasons that both bind well before that: `_from_solution`'s scale
# completion, `exp(-logΓ(1 + 1/shape))`, underflows to EXACTLY ZERO once
# `logΓ(1 + 1/shape)` exceeds roughly `-log(floatmin(T))` (~708 in
# Float64); and, tighter still, Distributions.jl's own `var(::Weibull)`
# computes `gamma(1 + 2/shape)` directly rather than in log-space, which
# overflows once `1 + 2/shape` exceeds roughly 171 in Float64 (measured:
# finite at shape = 0.0125, `Inf` by shape = 0.0117). `shape_min = 0.0125`
# keeps a comfortable margin above that measured edge while still reaching
# CVs many orders of magnitude past anything epidemiologically meaningful
# (see `_weibull_cv_max` below).
_weibull_shape_min(::Type{T}) where {T} = T(0.0125)
_weibull_shape_max(::Type{T}) where {T} = T(1e6)

function _moment_bracket(::Type{Weibull}, ::Val{(:mean, :sd)}, vals)
    T = eltype(vals)
    return log(_weibull_shape_min(T)), log(_weibull_shape_max(T))
end

# The shape follows from the solved root; the scale then follows in closed
# form, `b = mean / Γ(1 + 1/a) = mean * exp(-logΓ(1 + u))`.
function _from_solution(::Type{Weibull}, ::Val{(:mean, :sd)}, s, vals)
    mean, sd = vals
    a = exp(s)
    u = exp(-s)
    b = mean * exp(-loggamma(1 + u))
    return Weibull(a, b; check_args = false)
end

# Both bounds are the CV attained at the registered bracket's ends, so they
# are derived from the bracket rather than hand-tuned; a sampler exploring
# an unconstrained `sd` gets `-Inf` from this guard rather than reaching a
# request the bracket cannot answer.
#
# LOWER: a CV below the value attained at `shape_max` has no root inside
# the bracket. `pi / (sqrt(6) * shape_max)` is the large-shape asymptote
# `cv ~ pi / (sqrt(6) * shape)`, accurate to better than 1e-4 relative here.
_weibull_cv_min(::Type{T}) where {T} = T(pi) / (sqrt(T(6)) * _weibull_shape_max(T))

# UPPER: the CV attained at `shape_min`, computed via the exact relation
# the residual equation encodes (the small-shape asymptote used above for
# `cv_min` is markedly less accurate in this regime, and evaluating the
# exact form here costs nothing extra — a couple of `loggamma` calls, not a
# solve). `cv^2 = expm1(x)` with `x = logΓ(1 + 2/shape_min) -
# 2 logΓ(1 + 1/shape_min)` is mathematically exact, but `x` is large enough
# (~136 in Float64) that `exp(x)` overflows a narrower type — Float32's
# range ends around `exp(88)` — well before `sqrt` could bring the result
# back down to something representable. Halving the exponent first,
# `exp(x / 2) ≈ sqrt(expm1(x))` for `x` this large (the `-1` is
# astronomically negligible here), avoids that intermediate overflow.
function _weibull_cv_max(::Type{T}) where {T}
    a = _weibull_shape_min(T)
    x = loggamma(1 + 2 / a) - 2 * loggamma(1 + 1 / a)
    return exp(x / 2)
end

function _valid_moments(::Type{Weibull}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    (mean > 0 && sd > 0) || return false
    cv = sd / mean
    T = typeof(cv)
    return _weibull_cv_min(T) < cv < _weibull_cv_max(T)
end

# The same, given the variance instead of the standard deviation, matching
# every other family above.
function to_native(::Type{Weibull}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return to_native(Weibull, Val((:mean, :sd)), (mean, sqrt(var)))
end

function _valid_moments(::Type{Weibull}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return mean > 0 && var > 0 &&
           _valid_moments(Weibull, Val((:mean, :sd)), (mean, sqrt(var)))
end
