@testitem "test_reparameterisation covers every registered family" begin
    using Distributions, Roots
    using ReparameterisedDistributions: test_reparameterisation

    # Every registered (family, names) pair, with a valid point and the
    # invalid points its guard has to reject. `Roots` is loaded so the two
    # Weibull pairs, whose conversion is a root-find, have a solver.
    #
    # Whatever a later family adds, it belongs here too: this is the same
    # call an external package makes to check its own registration, so
    # running it over this package's own families keeps the two honest
    # against each other.
    cases = [
        (LogNormal, (:mean, :sd), (8.0, 2.0),
            ((8.0, -1.0), (-8.0, 1.0))),
        (LogNormal, (:mean, :var), (8.0, 4.0),
            ((8.0, -4.0), (-8.0, 4.0))),
        (Gamma, (:mean, :sd), (8.0, 3.0), ((8.0, -1.0), (-8.0, 1.0))),
        (Gamma, (:mean, :var), (8.0, 9.0), ((8.0, -9.0), (-8.0, 9.0))),
        (Gamma, (:mean, :shape), (8.0, 3.0), ((8.0, -3.0), (-8.0, 3.0))),
        (Gamma, (:rate, :shape), (0.5, 2.0), ((-0.5, 2.0), (0.5, -2.0))),
        (Exponential, (:rate,), (0.5,), ((-0.5,), (0.0,))),
        (NegativeBinomial, (:mean, :overdispersion), (10.0, 0.1),
            ((10.0, 0.0), (-10.0, 0.1))),
        (NegativeBinomial, (:dispersion, :mean), (2.0, 10.0),
            ((-2.0, 10.0), (2.0, -10.0))),
        (SkewNormal, (:centre, :mass_below_centre, :scale), (0.0, 0.3, 1.0),
            ((0.0, 0.0, 1.0), (0.0, 0.3, -1.0))),
        (Beta, (:mean, :sd), (0.3, 0.1), ((0.5, 0.5), (1.5, 0.1))),
        (Beta, (:mean, :var), (0.3, 0.01), ((0.5, 0.3), (1.5, 0.01))),
        (InverseGaussian, (:mean, :sd), (3.0, 2.0),
            ((3.0, -2.0), (-3.0, 2.0))),
        (InverseGaussian, (:mean, :var), (3.0, 4.0),
            ((3.0, -4.0), (-3.0, 4.0))),
        (Weibull, (:mean, :sd), (8.0, 3.0), ((8.0, -3.0), (-8.0, 3.0))),
        (Weibull, (:mean, :var), (8.0, 9.0), ((8.0, -9.0), (-8.0, 9.0)))
    ]

    # Guard the guard: one case per registered pair, so a family added
    # without a case here fails this count rather than going unchecked.
    @test length(cases) == 16

    for (D, names, vals, invalid) in cases
        test_reparameterisation(D, names, vals; invalid = invalid)
    end
end

@testitem "an out-of-package registration passes the same suite" begin
    using Distributions
    using ReparameterisedDistributions: test_reparameterisation

    # A registration made exactly as a downstream package makes one: add a
    # method to each of the two hooks, then check the result with the same
    # interface suite this package uses on its own families. `Laplace` is a
    # Distributions.jl family this package does not register, so nothing
    # here shadows a shipped method.
    #
    # `Laplace(mu, theta)` has mean `mu` and variance `2 * theta^2`, so the
    # location is native and `theta = sd / sqrt(2)`. The location is
    # unconstrained, so only the scale is guarded.
    function ReparameterisedDistributions.valid_moments(
            ::Type{Laplace}, ::Val{(:mean, :sd)}, vals)
        _, sd = vals
        return sd > 0
    end

    function ReparameterisedDistributions.to_native(
            ::Type{Laplace}, ::Val{(:mean, :sd)}, vals)
        m, sd = vals
        return Laplace(m, sd / sqrt(oftype(sd, 2)); check_args = false)
    end

    test_reparameterisation(Laplace, (:mean, :sd), (3.0, 2.0);
        invalid = ((3.0, -2.0), (3.0, 0.0)))
end

@testitem "test_reparameterisation catches a mis-registered pair" begin
    using Distributions, Test
    using ReparameterisedDistributions: test_reparameterisation

    # `test_reparameterisation` exists to catch two mistakes a valid point
    # alone never reveals: a registration reachable only under the wrong
    # (unsorted) name order, and a `to_native` whose return type admits
    # `nothing` (the #86 defect). Both are checked here by confirming the
    # suite actually fails on them, rather than by re-deriving the same
    # check the helper already makes.
    #
    # `test_reparameterisation` runs its own `@testset`, so calling it
    # directly would fail this testitem too. `probe_failures` isolates that
    # inner testset on its own stack frame and inspects its results
    # instead of letting them propagate, the same technique Test.jl's own
    # test suite uses to test `@testset` itself.
    function probe_failures(f)
        ts = Test.DefaultTestSet("probe")
        Test.push_testset(ts)
        ok = true
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                try
                    f()
                catch
                    ok = false
                end
            end
        end
        Test.pop_testset()
        return !ok || _has_failure(ts)
    end

    function _has_failure(ts::Test.DefaultTestSet)
        for r in ts.results
            if r isa Test.Fail || r isa Test.Error
                return true
            elseif r isa Test.DefaultTestSet && _has_failure(r)
                return true
            end
        end
        return false
    end

    # Guard the guard: the probe reports no failure for a pair already
    # known to pass the suite, so a probe that always answered `true`
    # would not slip past the two checks below undetected.
    @test !probe_failures() do
        test_reparameterisation(Gamma, (:mean, :sd), (8.0, 3.0);
            invalid = ((8.0, -1.0),))
    end

    # `Cauchy(location, scale)` is not registered by this package.
    # Register it under the wrong, unsorted name order, the mistake
    # `adding-a-reparameterisation.md` warns against: `reparameterise`
    # always sorts keywords before dispatching. Asking the suite to check
    # the canonical `(:location, :scale)` order finds no such method, only
    # the error-raising fallback, exactly as it would for every caller of
    # `reparameterise` on an uncanonically registered family.
    function ReparameterisedDistributions.valid_moments(
            ::Type{Cauchy}, ::Val{(:scale, :location)}, vals)
        scale, _ = vals
        return scale > 0
    end

    function ReparameterisedDistributions.to_native(
            ::Type{Cauchy}, ::Val{(:scale, :location)}, vals)
        scale, location = vals
        return Cauchy(location, scale; check_args = false)
    end

    @test probe_failures() do
        test_reparameterisation(Cauchy, (:location, :scale), (0.0, 1.0))
    end

    # `Laplace(mean, var)` is not registered by this package either.
    # `to_native` here is exactly the #86 shape: a branch that can return
    # `nothing`, so its inferred return type is a `Union` even though this
    # particular call (`var = 2.0 > 0`) never actually takes that branch.
    function ReparameterisedDistributions.valid_moments(
            ::Type{Laplace}, ::Val{(:mean, :var)}, vals)
        _, v = vals
        return v > 0
    end

    function ReparameterisedDistributions.to_native(
            ::Type{Laplace}, ::Val{(:mean, :var)}, vals)
        m, v = vals
        v <= 0 && return nothing
        return Laplace(m, sqrt(v / oftype(v, 2)); check_args = false)
    end

    @test probe_failures() do
        test_reparameterisation(Laplace, (:mean, :var), (3.0, 2.0))
    end
end

@testitem "No Test extension loaded: the stub's own error" begin
    using Distributions
    using ReparameterisedDistributions: test_reparameterisation

    # The realistic "Test not loaded" path cannot be produced in-process:
    # every test item here already has `Test` loaded, which triggers
    # `ReparameterisedDistributionsTestExt` regardless of who asked for
    # it, so the extension is always active by the time any test runs.
    # This asserts directly on the stub method instead, with a `names`
    # type (a `Vector`, not a `Tuple{Vararg{Symbol}}`) that only the
    # core's generic `(D, names, vals; kwargs...)` method, not the
    # extension's more specific one, can match. Mirrors the technique
    # already used for `_solve_moment_equation`'s stub in test/numeric.jl.
    @test_throws ArgumentError test_reparameterisation(
        LogNormal, [:mean, :sd], (8.0, 2.0))
    try
        test_reparameterisation(LogNormal, [:mean, :sd], (8.0, 2.0))
    catch e
        msg = sprint(showerror, e)
        @test occursin("Test", msg)
        @test occursin("ReparameterisedDistributionsTestExt", msg)
    end
end
