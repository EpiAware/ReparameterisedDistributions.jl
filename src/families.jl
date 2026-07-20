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
function _to_native(::Type{LogNormal}, ::Val{(:mean, :sd)}, vals)
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
function _to_native(::Type{LogNormal}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return _to_native(LogNormal, Val((:mean, :sd)), (mean, sqrt(var)))
end

function _valid_moments(::Type{LogNormal}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return mean > 0 && var > 0
end

# Gamma by mean and standard deviation. A Gamma(shape, scale) has
#   mean = shape * scale,  var = shape * scale^2
# so scale = var / mean and shape = mean / scale = mean^2 / var.
function _to_native(::Type{Gamma}, ::Val{(:mean, :sd)}, vals)
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

function _to_native(::Type{Gamma}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return _to_native(Gamma, Val((:mean, :sd)), (mean, sqrt(var)))
end

function _valid_moments(::Type{Gamma}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return mean > 0 && var > 0
end

# Gamma by mean and shape, which is how a delay is often elicited when the shape
# carries the meaning (a fixed number of exponential stages, say). The shape is
# native, so only the scale is derived: scale = mean / shape. This is the pair
# CensoredDistributions registered.
function _to_native(::Type{Gamma}, ::Val{(:mean, :shape)}, vals)
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
function _to_native(::Type{NegativeBinomial},
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
