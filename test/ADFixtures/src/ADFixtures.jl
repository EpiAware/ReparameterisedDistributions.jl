# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# AD-fixture registry implementing the EpiAwarePackageTools `ADRegistry`
# contract: scenarios (each with a ForwardDiff reference), a backend list, and
# broken/skip bookkeeping. The shared harness (driven from `test/ad/setup.jl`)
# consumes these.
#
# Every scenario differentiates with respect to the MOMENTS, which is the point
# of the package: the moments are the estimable parameters, so it is the
# gradient in moment coordinates that inference needs, and the closed-form
# conversion is what has to stay differentiable.
__precompile__(false)
module ADFixtures

using ADTypes: AutoForwardDiff, AutoReverseDiff, AutoMooncake,
               AutoMooncakeForward, AutoEnzyme
using DifferentiationInterface: DifferentiationInterface, Constant
import DifferentiationInterfaceTest as DIT
import ForwardDiff, ReverseDiff, Enzyme, Mooncake
using Distributions: Beta, Exponential, Gamma, InverseGaussian, LogNormal,
                     Logistic, NegativeBinomial, Normal, SkewNormal, Uniform,
                     Weibull, logpdf, cdf
using ReparameterisedDistributions: reparameterise

export scenarios, backends, broken_scenario_names,
       backend_broken_scenarios, backend_skip_scenarios

# ForwardDiff reference gradient for a scenario function.
function _reference(f, θ, contexts)
    return DifferentiationInterface.gradient(
        f, AutoForwardDiff(), θ, contexts...)
end

# Observations travel as a `Constant` context rather than a closure capture, so
# Enzyme differentiates cleanly.
const _OBS = [4.2, 7.1, 9.8, 12.4, 6.0]

# Counts, for the discrete family.
const _COUNTS = [3, 8, 12, 5, 21]

# Proportions, for Beta: its support is (0, 1), so `_OBS` does not fit.
const _PROPS = [0.12, 0.35, 0.28, 0.41, 0.19]

# `θ = [mean, sd]` — the coordinates a sampler actually moves in.
function _meansd_loglik(θ, obs)
    d = reparameterise(LogNormal; mean = θ[1], sd = θ[2], check_args = false)
    return sum(x -> logpdf(d, x), obs)
end

# The variance-named parameterisation, so the second registered name tuple is
# covered rather than assumed to behave.
function _meanvar_loglik(θ, obs)
    d = reparameterise(LogNormal; mean = θ[1], var = θ[2], check_args = false)
    return sum(x -> logpdf(d, x), obs)
end

# A `cdf` path as well as a density one: a censored or truncated likelihood
# differentiates through the cdf, and that path can break independently.
function _meansd_cdf(θ, obs)
    d = reparameterise(LogNormal; mean = θ[1], sd = θ[2], check_args = false)
    return sum(x -> cdf(d, x), obs)
end

function _gamma_meansd_loglik(θ, obs)
    d = reparameterise(Gamma; mean = θ[1], sd = θ[2], check_args = false)
    return sum(x -> logpdf(d, x), obs)
end

# The (mean, shape) pair derives only the scale, so it is a different code path
# from (mean, sd) and can break on its own.
function _gamma_meanshape_loglik(θ, obs)
    d = reparameterise(Gamma; mean = θ[1], shape = θ[2], check_args = false)
    return sum(x -> logpdf(d, x), obs)
end

# The discrete family. Counts are the observations, and the gradient is taken
# with respect to the mean and the overdispersion — a `1 / a` appears in the
# conversion, so this is the most fragile of the closed forms under AD.
function _nbinom_loglik(θ, counts)
    d = reparameterise(NegativeBinomial; mean = θ[1], overdispersion = θ[2],
        check_args = false)
    return sum(k -> logpdf(d, k), counts)
end

# The reciprocal convention. `θ = [dispersion, mean]`, the canonical (sorted)
# order the wrapper itself stores, so this exercises the parameterisation the
# way the closed form actually sees it rather than the call-site order.
function _nbinom_dispersion_loglik(θ, counts)
    d = reparameterise(NegativeBinomial; dispersion = θ[1], mean = θ[2],
        check_args = false)
    return sum(k -> logpdf(d, k), counts)
end

# `θ = [rate]` — a single-parameter family, so this also exercises a
# length-1 registered name tuple under AD.
function _exponential_rate_loglik(θ, obs)
    d = reparameterise(Exponential; rate = θ[1], check_args = false)
    return sum(x -> logpdf(d, x), obs)
end

# The reciprocal of (mean, shape): the rate, not the mean, is native-adjacent
# here (native scale = 1 / rate), so this is a different code path.
function _gamma_rateshape_loglik(θ, obs)
    d = reparameterise(Gamma; rate = θ[1], shape = θ[2], check_args = false)
    return sum(x -> logpdf(d, x), obs)
end

# `θ = [centre, mass_below_centre, scale]`, the canonical (sorted) order. Not a
# `cdf` scenario: Distributions.jl has no `cdf` for `SkewNormal` at all (Owen's
# T is not implemented there), so only the density path is exercised.
function _skewnormal_loglik(θ, obs)
    d = reparameterise(SkewNormal; centre = θ[1], mass_below_centre = θ[2],
        scale = θ[3], check_args = false)
    return sum(x -> logpdf(d, x), obs)
end

# `θ = [mean, sd]`. The conversion divides by `sd^2` twice over (once inside
# `nu`, once again through `alpha`/`beta`), so this is the fragile case among
# the new closed forms under AD.
function _beta_meansd_loglik(θ, props)
    d = reparameterise(Beta; mean = θ[1], sd = θ[2], check_args = false)
    return sum(x -> logpdf(d, x), props)
end

# `θ = [mean, sd]`. Unlike Gamma, the mean is already native here (the
# conversion only derives the shape), so this exercises a closed form that
# differentiates through a cube rather than a ratio of squares.
function _invgauss_meansd_loglik(θ, obs)
    d = reparameterise(InverseGaussian; mean = θ[1], sd = θ[2],
        check_args = false)
    return sum(x -> logpdf(d, x), obs)
end

# The one NUMERIC family: the conversion is a solver-backed root-find rather
# than exact algebra (see src/numeric.jl and the Weibull registration in
# src/families.jl), so this is the scenario that exercises the
# implicit-function-theorem correction under every wired backend, not just
# ForwardDiff.
function _weibull_meansd_loglik(θ, obs)
    d = reparameterise(Weibull; mean = θ[1], sd = θ[2], check_args = false)
    return sum(x -> logpdf(d, x), obs)
end

# `θ = [q05, q95]` — the elicited quantile VALUES are the estimable
# parameters here, and the probabilities are constants carried in the
# wrapper's type. The inversion solves a two-by-two linear system, so the
# gradient runs through a reciprocal determinant.
function _lognormal_quantiles_loglik(θ, obs)
    d = reparameterise(LogNormal; quantiles = (0.05 => θ[1], 0.95 => θ[2]),
        check_args = false)
    return sum(x -> logpdf(d, x), obs)
end

# The same inversion without the log transform, so the transform is not the
# only thing standing between the constraint values and the density.
function _normal_quantiles_loglik(θ, obs)
    d = reparameterise(Normal; quantiles = (0.25 => θ[1], 0.75 => θ[2]),
        check_args = false)
    return sum(x -> logpdf(d, x), obs)
end

# A mixed constraint set: `θ = [median, q95]`, in the canonical order the
# wrapper stores, so the moment row and the quantile row are differentiated
# side by side.
function _lognormal_mixed_loglik(θ, obs)
    d = reparameterise(LogNormal; median = θ[1], quantiles = (0.95 => θ[2],),
        check_args = false)
    return sum(x -> logpdf(d, x), obs)
end

# `θ = [q95]` — a one-constraint family, so the scale-only row is covered
# as well as the two-row solve.
function _exponential_quantile_loglik(θ, obs)
    d = reparameterise(Exponential; quantiles = (0.95 => θ[1],),
        check_args = false)
    return sum(x -> logpdf(d, x), obs)
end

# The `cdf` path for a quantile constraint set, which can break
# independently of the density path.
function _normal_quantiles_cdf(θ, obs)
    d = reparameterise(Normal; quantiles = (0.25 => θ[1], 0.75 => θ[2]),
        check_args = false)
    return sum(x -> cdf(d, x), obs)
end

# `θ = [mean, sd]` through the same constraint solve, reached by the
# concrete `(:mean, :sd)` registration rather than the catch-all. The
# `Uniform` scale carries a `sqrt(12)`, and its density is flat, so the
# gradient runs through the support's own edges.
function _uniform_meansd_loglik(θ, obs)
    d = reparameterise(Uniform; mean = θ[1], sd = θ[2], check_args = false)
    return sum(x -> logpdf(d, x), obs)
end

function _logistic_meansd_loglik(θ, obs)
    d = reparameterise(Logistic; mean = θ[1], sd = θ[2], check_args = false)
    return sum(x -> logpdf(d, x), obs)
end

"""
    scenarios(; with_reference = false, category = :marginal)

The AD gradient scenarios. Each is a `DIT.Scenario{:gradient, :out}` whose
`res1` carries a ForwardDiff reference when `with_reference = true`.
"""
function scenarios(; with_reference::Bool = false, category::Symbol = :marginal)
    out = DIT.Scenario{:gradient, :out}[]
    reals = (Constant(_OBS),)
    counts = (Constant(_COUNTS),)
    props = (Constant(_PROPS),)

    cases = (("LogNormal(mean, sd) loglik", _meansd_loglik, [8.0, 2.0], reals),
        ("LogNormal(mean, var) loglik", _meanvar_loglik, [8.0, 4.0], reals),
        ("LogNormal(mean, sd) cdf", _meansd_cdf, [8.0, 2.0], reals),
        ("Gamma(mean, sd) loglik", _gamma_meansd_loglik, [8.0, 3.0], reals),
        ("Gamma(mean, shape) loglik", _gamma_meanshape_loglik, [8.0, 3.0],
            reals),
        ("NegativeBinomial(mean, overdispersion) loglik", _nbinom_loglik,
            [10.0, 0.1], counts),
        ("NegativeBinomial(dispersion, mean) loglik",
            _nbinom_dispersion_loglik, [2.0, 10.0], counts),
        ("Exponential(rate) loglik", _exponential_rate_loglik, [0.5], reals),
        ("Gamma(rate, shape) loglik", _gamma_rateshape_loglik, [0.5, 3.0],
            reals),
        ("SkewNormal(centre, mass_below_centre, scale) loglik",
            _skewnormal_loglik, [8.0, 0.3, 2.0], reals),
        ("Beta(mean, sd) loglik", _beta_meansd_loglik, [0.3, 0.1], props),
        ("InverseGaussian(mean, sd) loglik", _invgauss_meansd_loglik,
            [3.0, 2.0], reals),
        ("Weibull(mean, sd) loglik", _weibull_meansd_loglik, [8.0, 3.0],
            reals),
        # Near the fragile edge of the numerically solvable CV window
        # (`_weibull_shape_max` in src/families.jl): `mean = 74.916, sd =
        # 1.079` is the point `solve_moment`'s own comments in
        # src/numeric.jl single out as one where a bracketed solve in
        # `Dual` arithmetic is unreliable even though the corrected
        # derivative is exact. It sits inside the solvable window, not on
        # its boundary — AT `cv_min` itself every backend (including
        # ForwardDiff) degrades to NaN/Inf, a separate, pre-existing shared
        # numerical edge unrelated to this scenario. Registered so a
        # regression at this edge is caught by the AD suite automatically,
        # rather than relying on a one-off manual check.
        ("Weibull(mean, sd) loglik (near CV window edge)",
            _weibull_meansd_loglik, [74.916, 1.079], reals),
        # `mean < 0`: `valid_moments == false`, so `logpdf` is the constant
        # `-Inf` short-circuit, never reaching `to_native`. No scenario
        # above reaches that branch under AD — where #86's bug lived. The
        # gradient is the zero vector on every backend, since the loglik is
        # locally constant. Not the CV-edge cases above, which are valid
        # but numerically fragile, a different failure mode.
        ("Gamma(mean, sd) loglik at invalid moments", _gamma_meansd_loglik,
            [-8.0, 3.0], reals),
        ("LogNormal(quantiles) loglik", _lognormal_quantiles_loglik,
            [1.2, 8.4], reals),
        ("Normal(quantiles) loglik", _normal_quantiles_loglik, [4.0, 10.0],
            reals),
        ("Normal(quantiles) cdf", _normal_quantiles_cdf, [4.0, 10.0], reals),
        ("LogNormal(median, quantiles) loglik", _lognormal_mixed_loglik,
            [4.0, 12.0], reals),
        ("Exponential(quantiles) loglik", _exponential_quantile_loglik,
            [20.0], reals),
        ("Uniform(mean, sd) loglik", _uniform_meansd_loglik, [8.0, 3.0],
            reals),
        ("Logistic(mean, sd) loglik", _logistic_meansd_loglik, [8.0, 3.0],
            reals))

    for (name, f, θ, contexts) in cases
        push!(out,
            DIT.Scenario{:gradient, :out}(f, θ, contexts...; name = name,
                # Prepare at the real point, not at `zero(x)`: a zero mean would
                # build a degenerate distribution and trip a domain error, and a
                # zero overdispersion would divide by zero.
                prep_args = (; x = θ, contexts = contexts),
                res1 = with_reference ? _reference(f, θ, contexts) : nothing))
    end
    return out
end

"""
    backends()

The AD backends to test, as `(; name, backend)` named tuples.
"""
function backends()
    return [
        (name = "ForwardDiff", backend = AutoForwardDiff()),
        (name = "ReverseDiff (tape)",
            backend = AutoReverseDiff(compile = false)),
        (name = "Enzyme forward",
            backend = AutoEnzyme(
                mode = Enzyme.set_runtime_activity(Enzyme.Forward))),
        (name = "Enzyme reverse",
            backend = AutoEnzyme(
                mode = Enzyme.set_runtime_activity(Enzyme.Reverse))),
        (name = "Mooncake reverse", backend = AutoMooncake(config = nothing)),
        (name = "Mooncake forward", backend = AutoMooncakeForward())
    ]
end

"Scenario names broken on every backend."
broken_scenario_names() = String[]

# The Weibull scenario is the one NUMERIC family (src/numeric.jl): its
# conversion runs a scalar root-find (Roots.jl, via
# `ReparameterisedDistributionsRootsExt`) rather than exact algebra.
# ForwardDiff and ReverseDiff were always unaffected (the
# implicit-function-theorem correction supplies the derivative regardless of
# how the root itself was found — see `_primal` in src/numeric.jl); Enzyme
# and Mooncake used to trace INTO Roots' own internals when building a
# derivative rule for the solve and fail there
# (`Enzyme.Compiler.IllegalTypeAnalysisException` for both Enzyme variants,
# an `ArgumentError` — "not permissible to bitcast to a differentiable type
# during AD" — for both Mooncake variants). `ReparameterisedDistributionsEnzymeExt`
# and `ReparameterisedDistributionsMooncakeExt` now give
# `_solve_moment_equation` the same `EnzymeRules.inactive` /
# `Mooncake.@zero_derivative` treatment `_primal` already gets, so the IFT
# correction supplies the derivative on all six backends; verified against
# the ForwardDiff reference at several (mean, sd) points including the CV
# window's edge, on both arm64 and linux/amd64.
"Per-backend broken scenario names (`Dict{String, Set{String}}`)."
function backend_broken_scenarios()
    return Dict{String, Set{String}}()
end

"Per-backend scenario names too unstable to run at all."
backend_skip_scenarios() = Dict{String, Set{String}}()

end # module ADFixtures
