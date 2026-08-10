# The numeric seam (src/numeric.jl) and its one registered family,
# `Weibull(mean, sd)` (src/families.jl). Unlike every other family in
# test/families.jl, this conversion is a solver-backed root-find rather than
# exact algebra, so these tests exercise the seam itself — `solve_moment`,
# the implicit-function-theorem correction, the three ways a solve fails
# cleanly — and not just the Weibull maths.
#
# `using Distributions` is enough to load the Roots extension: Distributions
# hard-depends on Roots, so `Base.loaded_modules` already contains it by the
# time any test file runs, and Julia's extension mechanism triggers on that
# regardless of who loaded it. No test here needs an explicit `using Roots`.

@testitem "Weibull(mean, sd): the extension actually loads" begin
    ext = Base.get_extension(
        ReparameterisedDistributions, :ReparameterisedDistributionsRootsExt)
    @test ext !== nothing
end

@testitem "Weibull(mean, sd): moment recovery sweep" begin
    using Distributions

    means = (0.5, 1.0, 8.0, 100.0)
    cvs = (0.02, 0.05, 0.1, 0.3, 0.5227, 1.0, 2.0, 5.0, 20.0)
    for m in means, cv in cvs

        d = reparameterise(Weibull; mean = m, sd = m * cv)
        @test mean(d)≈m rtol=1e-10
        @test std(d)≈m * cv rtol=1e-10
    end

    # Exact anchors: CV depends on the shape alone. cv = 1 is the
    # exponential (shape = 1); shape = 2 has a CV known in closed form.
    @test Distributions.shape(native(reparameterise(Weibull;
        mean = 8.0, sd = 8.0)))≈1.0 atol=1e-12
    @test Distributions.shape(native(reparameterise(Weibull;
        mean = 1.0, sd = 0.5227232008770631)))≈2.0 atol=1e-10
end

@testitem "Weibull(mean, var) delegates to (mean, sd)" begin
    using Distributions

    by_sd = reparameterise(Weibull; mean = 8.0, sd = 3.0)
    by_var = reparameterise(Weibull; mean = 8.0, var = 9.0)

    @test native(by_var) ≈ native(by_sd)
    @test params(by_var) == (8.0, 9.0)
end

@testitem "Weibull(mean, sd): the driver reaches Weibull's own to_native" begin
    using Distributions

    d = reparameterise(Weibull; mean = 8.0, sd = 3.0)
    nd = native(d)

    @test nd isa Weibull
    @test mean(d) ≈ mean(nd)
    @test std(d) ≈ std(nd)
    @test logpdf(d, 7.0) ≈ logpdf(nd, 7.0)
end

# --- A numeric family registers the same two hooks as an analytic one -----

@testitem "A numeric family needs valid_moments and to_native" begin
    using Distributions
    import ReparameterisedDistributions: solve_moment, to_native

    # A test-only registration on `Gamma(mean, sdnumeric)`, a pair with no
    # existing method, using the two-hook contract every family uses —
    # analytic or numeric alike: a `valid_moments` predicate, and a
    # `to_native` method that assumes it has already passed and calls
    # `solve_moment` inside its body. No trait, no extra registration.
    # Solved directly (shape = mean^2 / sd^2 is exact), so this is a driver
    # oracle rather than a test of solver correctness: the numeric answer
    # can be cross-checked against the true `Gamma(mean, sd)` closed form.
    _residual(s, vals) = s - log(vals[1]^2 / vals[2]^2)
    _deriv(s, vals) = one(s)
    _bracket(pvals) = (eltype(pvals)(-20.0), eltype(pvals)(20.0))

    function ReparameterisedDistributions.valid_moments(
            ::Type{Gamma}, ::Val{(:mean, :sdnumeric)}, vals)
        m, sd = vals
        return m > 0 && sd > 0
    end

    function ReparameterisedDistributions.to_native(
            ::Type{Gamma}, ::Val{(:mean, :sdnumeric)}, vals)
        m, sd = vals
        s = solve_moment(Gamma, Val((:mean, :sdnumeric)), _residual, _deriv,
            _bracket, vals)
        shape = exp(s)
        return Gamma(shape, m / shape; check_args = false)
    end

    for cv in (0.05, 0.2, 1.0, 3.0, 5.0)
        m, sd = 8.0, 8.0 * cv
        d = reparameterise(Gamma; mean = m, sdnumeric = sd)
        analytic = to_native(Gamma, Val((:mean, :sd)), (m, sd))
        @test native(d).α≈analytic.α rtol=1e-10
        @test native(d).θ≈analytic.θ rtol=1e-10
    end
end

@testitem "An unregistered pair still raises the analytical error" begin
    using Distributions

    @test_throws ArgumentError reparameterise(Weibull; foo = 1.0, bar = 2.0)
    try
        reparameterise(Weibull; foo = 1.0, bar = 2.0)
    catch e
        @test occursin("no reparameterisation", e.msg)
        @test occursin("registered", e.msg)
    end
end

# --- The edge of the numerically solvable CV window ---------------------

@testitem "Weibull(mean, sd): the valid CV domain round-trips at its edges" begin
    using Distributions

    cv_min = ReparameterisedDistributions._weibull_cv_min(Float64)
    cv_max = ReparameterisedDistributions._weibull_cv_max(Float64)

    # A real accuracy assertion, not just finiteness: `shape_max` is tuned
    # (see its definition in src/families.jl) so the recovered moments
    # match the request end to end, not merely so the solve terminates.
    # `rtol = 1e-6` is deliberately looser than the sweep test's `1e-10` —
    # honest given the flat-derivative conditioning right at this edge —
    # while still an order of magnitude tighter than the worst case
    # (`3.5e-9`) measured by sweeping 10,000 (mean, cv) pairs at this edge.
    for cv in (cv_min * 1.01, cv_max * 0.99)
        d = reparameterise(Weibull; mean = 1.0, sd = cv)
        @test mean(d)≈1.0 rtol=1e-6
        @test std(d)≈cv rtol=1e-6
    end
end

@testitem "Weibull(mean, sd): outside the CV domain raises DomainError" begin
    using Distributions

    cv_min = ReparameterisedDistributions._weibull_cv_min(Float64)
    cv_max = ReparameterisedDistributions._weibull_cv_max(Float64)

    for cv in (cv_min * 0.5, cv_max * 1.5)
        @test_throws DomainError reparameterise(Weibull; mean = 1.0, sd = cv)
        try
            reparameterise(Weibull; mean = 1.0, sd = cv)
        catch e
            @test occursin("Weibull", sprint(showerror, e))
        end
    end

    # mean <= 0 and sd <= 0, as for every other family.
    @test_throws DomainError reparameterise(Weibull; mean = -1.0, sd = 1.0)
    @test_throws DomainError reparameterise(Weibull; mean = 1.0, sd = -1.0)
    @test_throws DomainError reparameterise(Weibull; mean = 8.0, var = -1.0)
end

@testitem "Weibull(mean, sd): check_args = false gives -Inf, not a throw" begin
    using Distributions

    cv_max = ReparameterisedDistributions._weibull_cv_max(Float64)
    d = reparameterise(Weibull; mean = 1.0, sd = cv_max * 1.5,
        check_args = false)

    @test logpdf(d, 5.0) == -Inf
    @test pdf(d, 5.0) == 0.0

    # The `_restype` contract: an invalid point comes back as the
    # wrapper's own element type, so a Float32 wrapper returns `-Inf32`
    # rather than widening under AD.
    d32 = reparameterise(Weibull; mean = 1.0f0, sd = Float32(cv_max) * 1.5f0,
        check_args = false)
    @test logpdf(d32, 5.0f0) === -Inf32
    @test pdf(d32, 5.0f0) === 0.0f0
end

# --- The three ways a numeric conversion fails cleanly --------------------

@testitem "A bracket that never changes sign raises DomainError" begin
    using Distributions
    import ReparameterisedDistributions: solve_moment

    # `s^2 + 1` never crosses zero, so `_check_bracket` must catch this
    # before the solver ever runs.
    function ReparameterisedDistributions.to_native(
            ::Type{Gamma}, ::Val{(:badbracket,)}, vals)
        s = solve_moment(Gamma, Val((:badbracket,)),
            (s, vals) -> s^2 + 1, (s, vals) -> 2s,
            pvals -> (eltype(pvals)(-1.0), eltype(pvals)(1.0)), vals)
        return Gamma(exp(s), 1.0; check_args = false)
    end

    @test_throws DomainError reparameterise(Gamma; badbracket = 1.0)
    try
        reparameterise(Gamma; badbracket = 1.0)
    catch e
        msg = sprint(showerror, e)
        @test occursin("Gamma", msg)
        @test occursin("does not change sign", msg)
    end
end

@testitem "A mis-registered derivative raises DomainError from _check_solved" begin
    using Distributions
    import ReparameterisedDistributions: solve_moment

    # The bracket is valid and the root (s = 3) is exact, so the solver
    # itself finds it cleanly; the deliberately-wrong derivative (always
    # zero) sends the implicit-function-theorem correction to a
    # division-by-zero, landing far off the root.
    function ReparameterisedDistributions.to_native(
            ::Type{Gamma}, ::Val{(:badtol,)}, vals)
        s = solve_moment(Gamma, Val((:badtol,)),
            (s, vals) -> s - 3.0, (s, vals) -> 0.0,
            pvals -> (eltype(pvals)(0.0), eltype(pvals)(10.0)), vals)
        return Gamma(exp(s), 1.0; check_args = false)
    end

    @test_throws DomainError reparameterise(Gamma; badtol = 1.0)
    try
        reparameterise(Gamma; badtol = 1.0)
    catch e
        msg = sprint(showerror, e)
        @test occursin("did not converge", msg)
        @test occursin("residual", msg)
        @test occursin("tolerance", msg)
    end
end

@testitem "No solver backend loaded: the stub's own error" begin
    # The realistic "no extension loaded" path cannot be produced
    # in-process: `using Distributions` already loads Roots (it is a hard
    # Distributions dependency), which triggers
    # `ReparameterisedDistributionsRootsExt` regardless of who asked for
    # it, so the extension is always active by the time any test runs.
    # This asserts directly on the stub method instead, with argument
    # types (`Nothing, Nothing`) that only the core's generic
    # `(f, lo, hi)` method — not the extension's `(f, lo::Real, hi::Real)`
    # — can match.
    @test_throws ArgumentError ReparameterisedDistributions._solve_moment_equation(
        identity, nothing, nothing)
    try
        ReparameterisedDistributions._solve_moment_equation(
            identity, nothing, nothing)
    catch e
        msg = sprint(showerror, e)
        @test occursin("Roots", msg)
        @test occursin("ReparameterisedDistributionsRootsExt", msg)
    end
end

# --- AD: the anti-truncation regression tests -----------------------------

@testitem "Weibull(mean, sd): a Dual survives the conversion" begin
    using Distributions, ForwardDiff

    d = reparameterise(Weibull;
        mean = ForwardDiff.Dual(8.0, 1.0, 0.0),
        sd = ForwardDiff.Dual(3.0, 0.0, 1.0), check_args = false)
    nd = native(d)

    @test nd isa Weibull{<:ForwardDiff.Dual}
    shape_partials = ForwardDiff.partials(Distributions.shape(nd))
    scale_partials = ForwardDiff.partials(Distributions.scale(nd))
    # Both partials of both native parameters must be finite AND nonzero:
    # a solver that truncated a Dual to its primal would give a Dual with
    # all-zero partials here, the exact failure this test pins.
    @test all(isfinite, shape_partials)
    @test all(isfinite, scale_partials)
    @test all(!=(0), shape_partials)
    @test all(!=(0), scale_partials)
end

@testitem "Weibull(mean, sd): gradient matches central differences" begin
    using Distributions, ForwardDiff

    obs = [4.2, 7.1, 9.8, 12.4, 6.0]
    function loglik(θ)
        d = reparameterise(Weibull; mean = θ[1], sd = θ[2],
            check_args = false)
        return sum(x -> logpdf(d, x), obs)
    end

    for cv in (0.05, 0.15, 0.3, 0.6, 1.2, 2.5, 8.0)
        θ = [8.0, 8.0 * cv]
        g = ForwardDiff.gradient(loglik, θ)

        h = 1e-6
        cd = [
            (loglik([θ[1] + h, θ[2]]) - loglik([θ[1] - h, θ[2]])) / 2h,
            (loglik([θ[1], θ[2] + h]) - loglik([θ[1], θ[2] - h])) / 2h
        ]
        @test g≈cd rtol=1e-6

        # The regression test for the truncation trap: the uncorrected
        # implementation returns an sd-partial of exactly 0.0.
        @test g[2] != 0.0
    end
end

@testitem "Weibull(mean, sd): Hessian matches a central difference of the gradient" begin
    using Distributions, ForwardDiff

    obs = [4.2, 7.1, 9.8, 12.4, 6.0]
    function loglik(θ)
        d = reparameterise(Weibull; mean = θ[1], sd = θ[2],
            check_args = false)
        return sum(x -> logpdf(d, x), obs)
    end

    θ = [8.0, 3.0]
    H = ForwardDiff.hessian(loglik, θ)

    # A central difference OF THE GRADIENT, not of the log-likelihood
    # itself: this is what pins the two-step implicit correction. With one
    # step the off-diagonal is 21% wrong.
    h = 1e-6
    g(θ) = ForwardDiff.gradient(loglik, θ)
    cd_H = [(g([θ[1] + h, θ[2]])[1] - g([θ[1] - h, θ[2]])[1]) / 2h
            (g([θ[1] + h, θ[2]])[2] - g([θ[1] - h, θ[2]])[2]) / 2h
            (g([θ[1], θ[2] + h])[1] - g([θ[1], θ[2] - h])[1]) / 2h
            (g([θ[1], θ[2] + h])[2] - g([θ[1], θ[2] - h])[2]) / 2h]
    cd_H = reshape(cd_H, 2, 2)

    @test H≈cd_H rtol=1e-6
end

@testitem "Weibull(mean, sd): construction is fully type-inferred" begin
    using Distributions, Test

    @inferred reparameterise(Weibull; mean = 8.0, sd = 3.0)
    @inferred reparameterise(Weibull; mean = 8.0f0, sd = 3.0f0)

    # A Float32 wrapper must convert to a Float32 native distribution, not
    # silently widen — that would break the `_restype` contract that makes
    # `-Inf` come back as the tracked type.
    d32 = reparameterise(Weibull; mean = 8.0f0, sd = 3.0f0)
    @test native(d32) isa Weibull{Float32}
end

# --- Integration and cost --------------------------------------------------

@testitem "Weibull(mean, sd): rescale re-solves through _build" begin
    using Distributions

    d = reparameterise(Weibull; mean = 8.0, sd = 3.0)
    scaled = rescale(d, 2.0)

    # `rescale` scales the named parameter (`:mean` by default) and holds
    # every other registered parameter FIXED at its own value — `sd` here,
    # not the CV — matching every other family's `rescale` contract (see
    # test/rescale.jl).
    @test mean(scaled) ≈ 16.0
    @test params(scaled)[2] == params(d)[2]
    @test std(scaled) ≈ std(d)
end

@testitem "Weibull(mean, sd): the conversion is allocation-free" begin
    using Distributions

    d = reparameterise(Weibull; mean = 8.0, sd = 3.0)
    native(d) # warm up
    @test @allocated(native(d)) == 0
end

@testitem "Weibull(mean, sd): a benchmark guard on the conversion" begin
    using Distributions, BenchmarkTools

    d = reparameterise(Weibull; mean = 8.0, sd = 3.0)
    trial = @benchmark native($d) samples=200 evals=10

    # Measured ~2.4-2.7us with zero allocations; an upper bound rather
    # than parity with an analytic pair (~0.2us), which would be flaky.
    # Generous enough to absorb CI jitter while still catching a
    # regression to, say, an unbounded bisection.
    @test minimum(trial).time < 200_000.0 # ns
    @test trial.allocs == 0
end

@testitem "Weibull(mean, sd): show still prints the native distribution" begin
    using Distributions

    d = reparameterise(Weibull; mean = 8.0, sd = 3.0)
    @test occursin("Weibull", sprint(show, "text/plain", d))
    @test occursin("mean = 8.0", sprint(show, d))
end

@testitem "Weibull(mean, sd): batch loglikelihood matches per-observation" begin
    using Distributions

    d = reparameterise(Weibull; mean = 8.0, sd = 3.0)
    obs = [4.2, 7.1, 9.8, 12.4, 6.0]

    @test loglikelihood(d, obs) ≈ sum(x -> logpdf(d, x), obs)
end
