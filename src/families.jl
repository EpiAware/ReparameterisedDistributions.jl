# Closed-form conversions from a family's alternative parameters to its native
# ones. Each is exact algebra, so it is differentiable and adds no solver. The
# native distribution is built with `check_args = false`: validity is checked
# once at construction (see `_check_native`), and a sampler probing an invalid
# point should get `-Inf` from the density rather than an error mid-gradient.

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

# The moments must be checked in their own coordinates. The conversion squares
# `sd / mean`, so a negative standard deviation maps onto exactly the same valid
# native distribution as its positive counterpart: checking only the native
# distribution would let the wrapper report an `sd` it does not behave as. A
# log-normal is supported on the positives, so its mean is positive too.
function _check_moments(::Type{LogNormal}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    mean > 0 || throw(DomainError(mean,
        "the mean of a LogNormal must be positive"))
    sd > 0 || throw(DomainError(sd,
        "the standard deviation of a LogNormal must be positive"))
    return nothing
end

# The same, given the variance instead of the standard deviation.
function _to_native(::Type{LogNormal}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    return _to_native(LogNormal, Val((:mean, :sd)), (mean, sqrt(var)))
end

function _check_moments(::Type{LogNormal}, ::Val{(:mean, :var)}, vals)
    mean, var = vals
    var > 0 || throw(DomainError(var,
        "the variance of a LogNormal must be positive"))
    return _check_moments(LogNormal, Val((:mean, :sd)), (mean, sqrt(var)))
end
