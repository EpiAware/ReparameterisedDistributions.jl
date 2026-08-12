# # [Priors on moments](@id priors-on-moments)
#
# ## Introduction
#
# A delay can be elicited as a mean and a standard deviation, and you may
# want to put the prior straight onto those.
# Distributions.jl names a `Gamma` by its shape and scale, so doing that means
# either choosing priors for parameters you have no belief about, or writing
# the transform yourself.
# This tutorial shows what independent priors on the native parameters imply
# about the mean, and how [`reparameterise`](@ref) lets the prior be put on the
# mean directly.
#
# ### What are we going to do in this exercise
#
# 1. Put independent priors on a Gamma's shape and scale, and look at the
#    prior they imply on the mean.
# 2. Put the prior on the mean directly, and compare the two.
# 3. Fit a delay in moment coordinates and read the posterior off the chain.
#
# ### What might I need to know before starting
#
# This tutorial builds on the [Getting started](@ref getting-started) overview.
# It uses Turing.jl for the fit, with CairoMakie and AlgebraOfGraphics for the
# figures.

# ## Packages used
#
# CairoMakie and AlgebraOfGraphics are used for plotting only.

using ReparameterisedDistributions, Distributions
using Turing, Random
using CairoMakie, AlgebraOfGraphics, DataFramesMeta

CairoMakie.activate!(type = "png", px_per_unit = 2)
set_theme!(theme_latexfonts(); fontsize = 14)

Random.seed!(1)

# ## The prior implied by native parameters
#
# Suppose the delay is believed to have a mean near 8 days.
# A Gamma has no mean parameter, so writing that belief with native
# parameters means choosing priors for a shape and a scale.

shape_prior = truncated(Normal(2.0, 1.0); lower = 0.0)
scale_prior = truncated(Normal(4.0, 2.0); lower = 0.0)

# Each draw from those priors is a `Gamma`, and `mean` reads off the mean it
# implies.

n = 20_000
native = [Gamma(rand(shape_prior), rand(scale_prior)) for _ in 1:n]
implied_mean = mean.(native);

# `reparameterise` builds the same family from a mean and a standard
# deviation, so the prior can be put on the mean itself.

mean_prior = truncated(Normal(8.0, 3.0); lower = 0.0)
sd_prior = truncated(Normal(3.0, 1.0); lower = 0.0)

moments = [
    reparameterise(Gamma; mean = rand(mean_prior), sd = rand(sd_prior))
        for _ in 1:n
]
direct_mean = mean.(moments);

# Both branches call `mean` on a `Gamma`; only the coordinates the prior was
# written in differ.
# In the second, `mean` returns the value that was asked for.

d = reparameterise(Gamma; mean = 8.0, sd = 3.0)

(mean(d), std(d))

# Plotted together, the two priors on the mean differ.

priors = vcat(
    DataFrame(mean = implied_mean, source = "implied by shape and scale"),
    DataFrame(mean = direct_mean, source = "prior on the mean")
)

draw(
    data(@rsubset(priors, :mean < 40)) *
        mapping(:mean, color = :source) *
        AlgebraOfGraphics.density()
)

# The implied prior is skewed and wider, and puts mass on means outside the
# elicited belief.
# It can be brought closer by tuning the native priors, but only indirectly:
# the mean is a product of the two, so it is not something either prior sets
# on its own.

# ## Fitting both parameterisations
#
# The same data, fitted twice: once in native coordinates, once in moment
# coordinates.
# The delay below has a mean of 8 days and a standard deviation of 3.

truth = reparameterise(Gamma; mean = 8.0, sd = 3.0)
y = rand(truth, 300)

@model function native_model(y)
    shape ~ shape_prior
    scale ~ scale_prior
    y .~ Gamma(shape, scale)
end

# `check_args = false` turns off the construction-time validity check.
# `reparameterise`'s own docs recommend this whenever the call sits on
# the right of a `~`: a step-size search can probe values far from the
# posterior before warmup has calibrated a sane step, and the closed-form
# conversion is not guaranteed to stay well-behaved that far out. Leaving
# the check on risks a construction-time exception raised mid-gradient,
# which crashes the sampler outright rather than reporting a poor density
# for that point.

@model function moment_model(y)
    delay_mean ~ mean_prior
    delay_sd ~ sd_prior
    y .~ reparameterise(
        Gamma; mean = delay_mean, sd = delay_sd,
        check_args = false
    )
end

native_chain = sample(native_model(y), NUTS(), 1000; progress = false)
moment_chain = sample(moment_model(y), NUTS(), 1000; progress = false)

# ## Reading the posterior
#
# The moment fit reports the mean directly.

summarystats(moment_chain)

# The native fit reports a shape and a scale, so a posterior for the mean has
# to be reconstructed from the draws.

native_mean = vec(native_chain[:shape]) .* vec(native_chain[:scale]);

# That transform is one line for a `Gamma` and a different line for every
# other family, has to be redone for the standard deviation, and is applied
# to the wrong thing without complaint if it is wrong.
# It is also only half the problem: the prior in that fit was still the
# implied one plotted above, not the belief about the mean.

posteriors = vcat(
    DataFrame(mean = native_mean, fit = "native, transformed"),
    DataFrame(mean = vec(moment_chain[:delay_mean]), fit = "moment")
)

draw(
    data(posteriors) * mapping(:mean, color = :fit) *
        AlgebraOfGraphics.density() +
        data(DataFrame(mean = [8.0])) * mapping(:mean) *
        visual(VLines, color = :black, linestyle = :dash)
)

# Both recover the mean, because 300 observations swamp the prior.
# The difference is what each model let you say beforehand, and what each
# chain hands back without further work.

# ## What to do next
#
# - The [Getting started](@ref getting-started) overview lists every supported
#   parameterisation and how to register a new one.
# - The [Public API](@ref public-api) documents `reparameterise`, `rescale`,
#   `native` and `to_native`.
