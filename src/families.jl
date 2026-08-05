# Closed-form conversions from a family's alternative parameters to its native
# ones. Each is exact algebra, so it is differentiable and adds no solver. The
# native distribution is built with `check_args = false`; a method returns
# `nothing` instead of a distribution for an invalid point, checked before any
# other work, so an invalid point gives `-Inf` rather than an error raised
# mid-gradient.

# LogNormal by the mean and standard deviation of the distribution itself,
# rather than of its logarithm. Inverting the log-normal moments,
#   mean = exp(mu + sigma^2 / 2),  var = mean^2 * (exp(sigma^2) - 1)
# gives sigma^2 = log1p((sd / mean)^2) and mu = log(mean) - sigma^2 / 2.
# `log1p` keeps the small-`sd / mean` case accurate.
#
# A log-normal is supported on the positives, so its mean is positive. The
# standard deviation must be checked in its own coordinates: the conversion
# squares `sd / mean`, so a negative one would otherwise build exactly the
# same, valid native distribution as its positive counterpart.
function to_native(::Type{LogNormal}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    (mean > 0 && sd > 0) || return nothing
    s2 = log1p((sd / mean)^2)
    return LogNormal(log(mean) - s2 / 2, sqrt(s2); check_args = false)
end

# The same, given the variance instead of the standard deviation. Guards
# `var > 0` itself, before the `sqrt` below: relying on the delegated call to
# catch a negative variance would mean the `sqrt` runs first and throws.
function to_native(::Type{LogNormal}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    var > 0 || return nothing
    return to_native(LogNormal, Val((:mean, :sd)), (mean, sqrt(var)))
end

# Gamma by mean and standard deviation. A Gamma(shape, scale) has
#   mean = shape * scale,  var = shape * scale^2
# so scale = var / mean and shape = mean / scale = mean^2 / var.
#
# As for the LogNormal, the conversion squares `sd`, so the sign has to be
# checked here or a negative standard deviation would give a valid — and
# identical — native distribution.
function to_native(::Type{Gamma}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    (mean > 0 && sd > 0) || return nothing
    scale = sd^2 / mean
    return Gamma(mean / scale, scale; check_args = false)
end

function to_native(::Type{Gamma}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    var > 0 || return nothing
    return to_native(Gamma, Val((:mean, :sd)), (mean, sqrt(var)))
end

# Gamma by mean and shape, which is how a delay is often elicited when the shape
# carries the meaning (a fixed number of exponential stages, say). The shape is
# native, so only the scale is derived: scale = mean / shape. This is the pair
# CensoredDistributions registered.
function to_native(::Type{Gamma}, ::Val{(:mean, :shape)}, vals)
    mean, shape = vals
    (mean > 0 && shape > 0) || return nothing
    return Gamma(shape, mean / shape; check_args = false)
end

# NegativeBinomial by mean and overdispersion, the parameterisation epidemiology
# reaches for: the overdispersion `a` (a cluster factor) is the excess variance
# relative to a Poisson, through
#   var = mean + a * mean^2
# so a -> 0 recovers the Poisson limit and larger `a` means more clustering. The
# native `NegativeBinomial(r, p)` has mean = r(1-p)/p and var = mean/p, giving
#   r = 1 / a,  p = mean / var = 1 / (1 + a * mean).
#
# `a = 0` is the Poisson limit, not a NegativeBinomial: `r = 1 / a` diverges,
# so it is rejected alongside `a < 0`.
#
# Note the family is DISCRETE. The wrapper takes its value support from the
# family, so this stays a discrete distribution rather than silently becoming a
# continuous one.
function to_native(::Type{NegativeBinomial},
        ::Val{(:mean, :overdispersion)}, vals)
    mean, a = vals
    (mean > 0 && a > 0) || return nothing
    p = 1 / (1 + a * mean)
    return NegativeBinomial(1 / a, p; check_args = false)
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
    (dispersion > 0 && mean > 0) || return nothing
    p = dispersion / (dispersion + mean)
    return NegativeBinomial(dispersion, p; check_args = false)
end

# Exponential by its rate, the quantity a hazard is usually written over and
# the quantity one wants reported, rather than the native scale (which is
# already the mean, but is still the reciprocal of the rate a prior is
# typically placed on). The native `Exponential(θ)` takes the scale directly,
# so the conversion is a single inversion: θ = 1 / rate.
function to_native(::Type{Exponential}, ::Val{(:rate,)}, vals)
    rate, = vals
    rate > 0 || return nothing
    return Exponential(1 / rate; check_args = false)
end

# Gamma by shape and rate, the reciprocal of the (mean, shape) pair above: the
# shape is native either way, and here the rate is native too once inverted to
# a scale, rather than the mean being. `scale = 1 / rate`.
#
# The canonical (sorted) name order is `(:rate, :shape)`, not
# `(:shape, :rate)`: `_canonical` sorts alphabetically, and 'r' < 's'.
function to_native(::Type{Gamma}, ::Val{(:rate, :shape)}, vals)
    rate, shape = vals
    (rate > 0 && shape > 0) || return nothing
    return Gamma(shape, 1 / rate; check_args = false)
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
    (scale > 0 && 0 < m < 1) || return nothing
    alpha = tan(pi * (1 / 2 - m))
    return SkewNormal(centre, scale, alpha; check_args = false)
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
    (0 < mean < 1 && sd > 0 && sd^2 < mean * (1 - mean)) || return nothing
    nu = mean * (1 - mean) / sd^2 - 1
    return Beta(mean * nu, (1 - mean) * nu; check_args = false)
end

function to_native(::Type{Beta}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    var > 0 || return nothing
    return to_native(Beta, Val((:mean, :sd)), (mean, sqrt(var)))
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
    (mean > 0 && sd > 0) || return nothing
    lambda = mean^3 / sd^2
    return InverseGaussian(mean, lambda; check_args = false)
end

function to_native(::Type{InverseGaussian}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    var > 0 || return nothing
    return to_native(InverseGaussian, Val((:mean, :sd)), (mean, sqrt(var)))
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
# family has no algebraic shortcut for the shape, so its `to_native` calls
# `solve_moment` (numeric.jl) instead of exact algebra.
#
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
function _weibull_residual(s, vals)
    mean, sd = vals
    u = exp(-s)
    return loggamma(1 + 2u) - 2 * loggamma(1 + u) - log1p((sd / mean)^2)
end

function _weibull_deriv(s, vals)
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
# reasons that both bind well before that: `to_native`'s scale
# completion, `exp(-logΓ(1 + 1/shape))`, underflows to EXACTLY ZERO once
# `logΓ(1 + 1/shape)` exceeds roughly `-log(floatmin(T))` (~708 in
# Float64); and, tighter still, Distributions.jl's own `var(::Weibull)`
# computes `gamma(1 + 2/shape)` directly rather than in log-space, which
# overflows once `1 + 2/shape` exceeds roughly 171 in Float64 (measured:
# finite at shape = 0.0125, `Inf` by shape = 0.0117). `shape_min = 0.0125`
# keeps a comfortable margin above that measured edge while still reaching
# CVs many orders of magnitude past anything epidemiologically meaningful
# (see `_weibull_cv_max` below).
#
# `shape_max` is tighter than the residual equation's own overflow bound
# for a reason found only by execution: `dF/ds -> 0` as `shape -> Inf` (the
# CV-vs-shape curve flattens there), so a solve whose RESIDUAL passes
# `_check_solved`'s tolerance can still sit on a shape whose actual CV
# misses the request by far more than that tolerance implies. Measured
# directly, cross-checked against an independent BigFloat solve of the
# same equation: at `shape_max = 1e6`, `mean = 1.0, sd = cv_min * 1.001`
# recovers `std(d)` with a relative error of `7e-5`, not the
# `sqrt(eps(Float64)) ~ 1.5e-8` the residual check promises. Swept over
# 10,000 (mean, cv) pairs at the domain's own edge, `shape_max = 5_000`
# keeps the worst recovered `std` relative error at `3.5e-9`, safely
# inside that promised tolerance end to end, while still reaching CVs down
# to about `2.6e-4` — many orders of magnitude past anything
# epidemiologically meaningful. The equivalent issue does not arise at
# `shape_min`: `dF/ds` grows, not shrinks, as `shape -> 0`.
_weibull_shape_min(::Type{T}) where {T} = T(0.0125)
_weibull_shape_max(::Type{T}) where {T} = T(5_000.0)

function _weibull_bracket(pvals)
    T = eltype(pvals)
    return log(_weibull_shape_min(T)), log(_weibull_shape_max(T))
end

# Both bounds are the CV attained at the registered bracket's ends, so they
# are derived from the bracket rather than hand-tuned; a sampler exploring
# an unconstrained `sd` gets `-Inf` from this guard rather than reaching a
# request the bracket cannot answer.
#
# LOWER: a CV below the value attained at `shape_max` has no root inside
# the bracket. `pi / (sqrt(6) * shape_max)` is the large-shape asymptote
# `cv ~ pi / (sqrt(6) * shape)`, accurate to better than 1e-4 relative here.
function _weibull_cv_min(::Type{T}) where {T}
    return T(pi) / (sqrt(T(6)) * _weibull_shape_max(T))
end

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

# The shape follows from the solved root; the scale then follows in closed
# form, `b = mean / Γ(1 + 1/a) = mean * exp(-logΓ(1 + u))`. The CV window
# is checked before the solve runs, so an out-of-window request is `nothing`
# rather than reaching `_check_bracket`'s throw.
function to_native(::Type{Weibull}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    (mean > 0 && sd > 0) || return nothing
    cv = sd / mean
    T = typeof(cv)
    (_weibull_cv_min(T) < cv < _weibull_cv_max(T)) || return nothing
    s = solve_moment(Weibull, Val((:mean, :sd)), _weibull_residual,
        _weibull_deriv, _weibull_bracket, vals)
    u = exp(-s)
    return Weibull(exp(s), mean * exp(-loggamma(1 + u)); check_args = false)
end

# The same, given the variance instead of the standard deviation, matching
# every other family above.
function to_native(::Type{Weibull}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    var > 0 || return nothing
    return to_native(Weibull, Val((:mean, :sd)), (mean, sqrt(var)))
end
