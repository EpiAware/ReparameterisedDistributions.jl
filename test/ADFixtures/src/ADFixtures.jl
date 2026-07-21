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
using Distributions: Exponential, Gamma, LogNormal, NegativeBinomial, logpdf,
                     cdf
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

"""
    scenarios(; with_reference = false, category = :marginal)

The AD gradient scenarios. Each is a `DIT.Scenario{:gradient, :out}` whose
`res1` carries a ForwardDiff reference when `with_reference = true`.
"""
function scenarios(; with_reference::Bool = false, category::Symbol = :marginal)
    out = DIT.Scenario{:gradient, :out}[]
    reals = (Constant(_OBS),)
    counts = (Constant(_COUNTS),)

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

"Per-backend broken scenario names (`Dict{String, Set{String}}`)."
backend_broken_scenarios() = Dict{String, Set{String}}()

"Per-backend scenario names too unstable to run at all."
backend_skip_scenarios() = Dict{String, Set{String}}()

end # module ADFixtures
