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
