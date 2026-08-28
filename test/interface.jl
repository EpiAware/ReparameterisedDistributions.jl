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
        (Weibull, (:mean, :var), (8.0, 9.0), ((8.0, -9.0), (-8.0, 9.0))),
        # The quantile constraint sets. A probability stands where a
        # `Symbol` would, and the invalid points are a decreasing pair,
        # which describes no distribution rather than a hard solve.
        (Normal, (0.25, 0.75), (1.0, 3.0), ((3.0, 1.0), (1.0, 1.0))),
        (Logistic, (0.25, 0.75), (1.0, 3.0), ((3.0, 1.0),)),
        (Cauchy, (0.25, 0.75), (1.0, 3.0), ((3.0, 1.0),)),
        (Laplace, (0.25, 0.75), (1.0, 3.0), ((3.0, 1.0),)),
        (Uniform, (0.25, 0.75), (1.0, 3.0), ((3.0, 1.0),)),
        (LogNormal, (0.05, 0.95), (1.2, 8.4), ((8.4, 1.2), (-1.2, 8.4))),
        (Exponential, (0.95,), (3.0,), ((-3.0,), (0.0,))),
        (LogNormal, (:median, 0.95), (4.0, 12.0), ((12.0, 4.0),)),
        (Normal, (:mean, 0.95), (8.0, 20.0), ((20.0, 8.0),)),
        (Normal, (:sd, 0.95), (2.0, 20.0), ((-2.0, 20.0),)),
        # The standard moments, where a location and a scale are what they
        # already are. Registered concretely so they stay unambiguous
        # against the mirror generic in src/standard_moments.jl.
        (Normal, (:mean, :sd), (5.0, 2.0), ((5.0, -2.0), (5.0, 0.0))),
        (Normal, (:mean, :var), (5.0, 4.0), ((5.0, -4.0),)),
        (Logistic, (:mean, :sd), (5.0, 2.0), ((5.0, -2.0),)),
        (Logistic, (:mean, :var), (5.0, 4.0), ((5.0, -4.0),)),
        (Laplace, (:mean, :sd), (5.0, 2.0), ((5.0, -2.0),)),
        (Laplace, (:mean, :var), (5.0, 4.0), ((5.0, -4.0),)),
        (Uniform, (:mean, :sd), (5.0, 2.0), ((5.0, -2.0),)),
        (Uniform, (:mean, :var), (5.0, 4.0), ((5.0, -4.0),)),
        (Exponential, (:mean,), (4.0,), ((-4.0,), (0.0,)))
    ]

    # Guard the guard: one case per registered pair, so a family added
    # without a case here fails this count rather than going unchecked.
    @test length(cases) == 35

    for (D, names, vals, invalid) in cases
        test_reparameterisation(D, names, vals; invalid = invalid)
    end
end

@testitem "an out-of-package registration passes the same suite" begin
    using Distributions
    using ReparameterisedDistributions: test_reparameterisation

    # A registration made exactly as a downstream package makes one: add a
    # method to each of the two hooks, then check the result with the same
    # interface suite this package uses on its own families. `Gumbel` is a
    # Distributions.jl family this package registers by nothing at all, so
    # the pair below shadows no shipped method. `Laplace` would not do:
    # it is in `_LinearMoments`, so this package already registers it by
    # `(:mean, :sd)` and the example would be shadowing rather than adding.
    #
    # `Gumbel(mu, theta)` has mean `mu + theta * eulergamma` and variance
    # `theta^2 * pi^2 / 6`, so both native parameters are derived. The
    # location is unconstrained, so only the scale is guarded.
    function ReparameterisedDistributions.valid_moments(
            ::Type{Gumbel}, ::Val{(:mean, :sd)}, vals)
        _, sd = vals
        return sd > 0
    end

    function ReparameterisedDistributions.to_native(
            ::Type{Gumbel}, ::Val{(:mean, :sd)}, vals)
        m, sd = vals
        theta = sd * sqrt(oftype(sd, 6)) / oftype(sd, pi)
        return Gumbel(m - theta * oftype(sd, Base.MathConstants.eulergamma),
            theta; check_args = false)
    end

    test_reparameterisation(Gumbel, (:mean, :sd), (3.0, 2.0);
        invalid = ((3.0, -2.0), (3.0, 0.0)))
end

@testitem "the checks test_reparameterisation relies on catch a bad pair" begin
    using Distributions
    using ReparameterisedDistributions: to_native

    # `test_reparameterisation` exists to catch two mistakes a valid point
    # alone never reveals: a registration reachable only under the wrong
    # (unsorted) name order, and a `to_native` whose return type admits
    # `nothing` (the #86 defect). This checks the two predicates the
    # helper itself uses to catch them (dispatch reachability via
    # `which`, and `Nothing`-admission via `Base.return_types`), on a
    # deliberately mis-registered pair, rather than running the full
    # helper and inspecting whether it failed.
    #
    # Running the helper and catching its own `@test` failures was tried
    # first, isolating them on a private `Test.DefaultTestSet` via
    # `Test.push_testset`/`Test.pop_testset`. Those two are unexported
    # internals, and Julia's `@testset` macro itself has no public way to
    # ask "did this nested block fail" without them: a custom
    # `AbstractTestSet` needs the very same internals to record itself
    # into its parent (see `Test.finish`'s own docstring), and a nested,
    # type-inheriting `@testset` (Julia always gives an untyped nested
    # `@testset` its parent's type) means every `@test` inside
    # `test_reparameterisation` would need that propagation to be
    # visible at all. That approach broke under a Julia prerelease whose
    # `Test` had renamed or dropped `push_testset`. Asserting the
    # predicates directly needs nothing from `Test` beyond ordinary
    # `@test`, so it cannot break the same way again.

    # Guard the guard: a pair already known to pass the suite should
    # pass both predicates directly, too.
    let argtypes = (Type{Gamma}, Val{(:mean, :sd)}, Tuple{Float64, Float64})
        fallback = which(to_native,
            (Type{Gamma}, Val{(:_unregistered_,)}, Tuple{Float64, Float64}))
        @test which(to_native, argtypes) !== fallback
        rt = only(Base.return_types(to_native, argtypes))
        @test !(Nothing <: rt)
    end

    # `Pareto(shape, scale)` is not registered by this package at all, not
    # even by quantile constraints, so nothing stands between it and the
    # fallback. Register it under the wrong, unsorted name order, the
    # mistake `adding-a-reparameterisation.md` warns against:
    # `reparameterise` always sorts keywords before dispatching. Asking for
    # the canonical `(:scale, :shape)` order finds no such method, only the
    # error-raising fallback, exactly as it would for every caller of
    # `reparameterise` on an uncanonically registered family.
    function ReparameterisedDistributions.valid_moments(
            ::Type{Pareto}, ::Val{(:shape, :scale)}, vals)
        shape, scale = vals
        return shape > 0 && scale > 0
    end

    function ReparameterisedDistributions.to_native(
            ::Type{Pareto}, ::Val{(:shape, :scale)}, vals)
        shape, scale = vals
        return Pareto(shape, scale; check_args = false)
    end

    let argtypes = (Type{Pareto}, Val{(:scale, :shape)},
            Tuple{Float64, Float64})
        fallback = which(to_native,
            (Type{Pareto}, Val{(:_unregistered_,)}, Tuple{Float64, Float64}))
        @test which(to_native, argtypes) === fallback
    end

    # `Gumbel(mean, var)` is not registered by this package either — and
    # `Laplace(mean, var)` no longer qualifies, since `_LinearMoments`
    # covers it. `to_native` here is exactly the #86 shape: a branch that
    # can return `nothing`, so its inferred return type is a `Union` even
    # though this particular call (`var = 2.0 > 0`) never takes that branch.
    function ReparameterisedDistributions.valid_moments(
            ::Type{Gumbel}, ::Val{(:mean, :var)}, vals)
        _, v = vals
        return v > 0
    end

    function ReparameterisedDistributions.to_native(
            ::Type{Gumbel}, ::Val{(:mean, :var)}, vals)
        m, v = vals
        v <= 0 && return nothing
        theta = sqrt(v) * sqrt(oftype(v, 6)) / oftype(v, pi)
        return Gumbel(m - theta * oftype(v, Base.MathConstants.eulergamma),
            theta; check_args = false)
    end

    let argtypes = (Type{Gumbel}, Val{(:mean, :var)},
            Tuple{Float64, Float64})
        rt = only(Base.return_types(to_native, argtypes))
        @test Nothing <: rt
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
