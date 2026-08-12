# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Per-backend AD gradient test items. Each backend is its own `@testitem`,
# tagged so the per-backend CI can select it with a tag filter (e.g.
# `julia test/ad/runtests.jl enzyme_reverse`). The harness wiring lives in the
# managed `setup.jl`; the SCENARIOS come from the package's own `ADFixtures`
# registry. This starter seed is generated from `_AD_BACKENDS` (the kit's
# single source of truth for the AD infra) at scaffold time, so it covers
# every backend the kit knows about; add/trim backends and categories to
# match the package afterwards (this file is write-once).

@testitem "ForwardDiff gradients (marginal)" tags = [:ad, :forwarddiff] setup = [ADHelpers] begin
    test_working_backend("ForwardDiff")
end

@testitem "ReverseDiff (tape) gradients (marginal)" tags = [:ad, :reversediff] setup = [ADHelpers] begin
    test_working_backend("ReverseDiff (tape)")
end

@testitem "ReverseDiff (compiled) gradients (marginal)" tags = [
    :ad, :reversediff_compiled,
] setup = [ADHelpers] begin
    test_working_backend("ReverseDiff (compiled)")
end

@testitem "Enzyme forward gradients (marginal)" tags = [:ad, :enzyme, :enzyme_forward] setup = [ADHelpers] begin
    test_working_backend("Enzyme forward")
end

@testitem "Enzyme reverse gradients (marginal)" tags = [:ad, :enzyme, :enzyme_reverse] setup = [ADHelpers] begin
    test_working_backend("Enzyme reverse")
end

@testitem "Mooncake reverse gradients (marginal)" tags = [:ad, :mooncake, :mooncake_reverse] setup = [ADHelpers] begin
    test_working_backend("Mooncake reverse")
end

@testitem "Mooncake forward gradients (marginal)" tags = [:ad, :mooncake, :mooncake_forward] setup = [ADHelpers] begin
    test_working_backend("Mooncake forward")
end

# Add latent (or other) scenario groups as the package needs, e.g.:
# @testitem "ForwardDiff gradients (latent)" tags=[:ad, :forwarddiff] setup=[ADHelpers] begin
#     test_working_backend("ForwardDiff"; category = :latent)
# end

# Regression marker for a known SECOND-ORDER limitation of the
# `EnzymeRules.inactive` shield on `_solve_moment_equation`
# (`ReparameterisedDistributionsEnzymeExt`): it is a first-order-only
# rule, and a Hessian through the Weibull numeric path throws a
# catchable Enzyme codegen `AssertionError` on both modes. Covers BOTH
# Enzyme modes in one item (rather than splitting per-tag) so either
# per-backend CI job (`enzyme_forward`/`enzyme_reverse`) still runs it.
#
# Deliberately NOT testing the Mooncake side here or anywhere else in
# this package: `SecondOrder(AutoForwardDiff(), AutoMooncake())` over
# this same path does not raise a catchable Julia exception — it
# ABORTS THE WHOLE PROCESS (`Abort trap: 6`, an LLVM assertion inside
# Mooncake's Enzyme-derived rule compiler; see
# `ReparameterisedDistributionsMooncakeExt`'s comment). `@test_broken`
# cannot catch a process abort, so exercising that combination here
# would take down this CI job with no diagnostic. It is documented,
# never executed.
@testitem "Weibull(mean, sd) Hessian is broken under Enzyme (documented)" tags = [
    :ad, :enzyme, :enzyme_forward, :enzyme_reverse,
] setup = [ADHelpers] begin
    f(θ) = ADFixtures._weibull_meansd_loglik(θ, ADFixtures._OBS)
    θ = [8.0, 3.0]
    for mode in (Enzyme.Forward, Enzyme.Reverse)
        backend = ADTypes.AutoEnzyme(mode = Enzyme.set_runtime_activity(mode))
        @test_broken DifferentiationInterface.hessian(f, backend, θ) isa
            AbstractMatrix
    end
end
