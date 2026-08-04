# # [Priors on moments](@id priors-on-moments)
#
# ## Introduction
#
# A delay can be elicited as a mean and a standard deviation.
# Distributions.jl names a `Gamma` by its shape and scale, so the coordinates
# the elicitation arrives in are not the ones the model has to be written in.
# The usual workaround is to put independent priors on the native parameters
# and hope the implied prior on the mean is close to what was meant.
# This tutorial shows why that hope is misplaced, and what
# [`reparameterise`](@ref) does instead.
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

# ## The prior you write is not the prior you meant
#
# Suppose the delay is believed to have a mean near 8 days.
# Writing that belief with native parameters means choosing priors for a shape
# and a scale, because a Gamma has no mean parameter to attach a prior to.
# Independent, individually reasonable choices look like this.

shape_prior = truncated(Normal(2.0, 1.0); lower = 0.0)
scale_prior = truncated(Normal(4.0, 2.0); lower = 0.0)

# The mean of a Gamma is the product of its shape and scale.
# So the prior on the mean is the pushforward of that product.

n = 20_000
implied_mean = rand(shape_prior, n) .* rand(scale_prior, n)

# Putting the prior where it was meant to go needs no pushforward at all.

mean_prior = truncated(Normal(8.0, 3.0); lower = 0.0)
direct_mean = rand(mean_prior, n)

# Plotted together, the two disagree about the thing the modeller actually had
# an opinion on.

priors = vcat(
    DataFrame(mean = implied_mean, source = "implied by shape × scale"),
    DataFrame(mean = direct_mean, source = "prior on the mean")
)

draw(
    data(@rsubset(priors, :mean < 40)) *
    mapping(:mean, color = :source) *
    AlgebraOfGraphics.density()
)

# The implied prior is skewed, wider, and puts real mass on means the modeller
# would have ruled out.
#
# The two priors also cannot be reconciled by tuning the native ones.
# Independent priors on shape and scale never compose into an arbitrary prior
# on the mean, because the mean is a product of the two and carries their
# dependence with it.

# ## Writing the model in moment coordinates
#
# `reparameterise` returns a distribution whose parameters are the moments, so
# the prior goes where it was elicited.
# The delay below has a mean of 8 days and a standard deviation of 3.

truth = reparameterise(Gamma; mean = 8.0, sd = 3.0)
y = rand(truth, 300)

@model function delay(y)
    delay_mean ~ truncated(Normal(8.0, 3.0); lower = 0.0)
    delay_sd ~ truncated(Normal(3.0, 2.0); lower = 0.0)
    y .~ reparameterise(Gamma; mean = delay_mean, sd = delay_sd,
        check_args = false)
end

# `check_args = false` matters inside a model.
# A sampler exploring an unconstrained space will propose a negative standard
# deviation, and turning the check off means that proposal scores `-Inf`
# instead of raising part-way through a gradient evaluation.

chain = sample(delay(y), NUTS(), 1000; progress = false)

# ## Reading the posterior
#
# The chain comes back in the coordinates the delay was elicited in, so the
# summary is directly comparable to what was believed beforehand.

summarystats(chain)

# Plotted against the values the data were generated from, both moments are
# recovered.

draws = vcat(
    DataFrame(value = vec(chain[:delay_mean]), moment = "mean"),
    DataFrame(value = vec(chain[:delay_sd]), moment = "sd")
)
actual = DataFrame(moment = ["mean", "sd"], value = [8.0, 3.0])

draw(
    data(draws) * mapping(:value, layout = :moment) *
    AlgebraOfGraphics.density() +
    data(actual) * mapping(:value, layout = :moment) *
    visual(VLines, color = :black, linestyle = :dash);
    facet = (; linkxaxes = :none)
)

# The conversion from moments to native parameters is exact algebra rather than
# a numerical solve, so the model stays differentiable and the gradient with
# respect to the mean and the standard deviation is exact.
# That is what lets NUTS sample in moment coordinates at no extra cost.

# ## What to do next
#
# - The [Getting started](@ref getting-started) overview lists every supported
#   parameterisation and how to register a new one.
# - The [Public API](@ref public-api) documents `reparameterise`, `rescale`,
#   `native` and `to_native`.
