# The standard-moment fallback (src/standard_moments.jl): a family with no
# registered closed form is converted by solving its own moment equations
# for its native parameters. These tests exercise the fallback itself — the
# solve, the domains hook, the arity rule and the guard — as well as the
# thing it must NOT do, which is take a registered closed form's place.

@testitem "an unregistered family converts from (mean, sd)" begin
    using Distributions

    # `Frechet` is not registered by this package, so this reaches the
    # generic numeric fallback rather than any closed form.
    d = reparameterise(Frechet; mean = 8.0, sd = 3.0)

    @test native(d) isa Frechet
    @test mean(d)≈8.0 rtol=1e-8
    @test std(d)≈3.0 rtol=1e-8
end

@testitem "the fallback recovers the moments across families" begin
    using Distributions

    # A family per shape of native parameter space: shape-and-scale,
    # location-and-scale, bounded, and one whose fields are not its
    # parameters (`InverseGamma` holds a `Gamma`).
    cases = ((Frechet, 8.0, 3.0), (Frechet, 8.0, 30.0), (Pareto, 8.0, 3.0),
        (InverseGamma, 8.0, 3.0), (BetaPrime, 2.0, 3.0),
        (BetaPrime, 1.0, 0.05), (Kumaraswamy, 0.3, 0.1),
        (Normal, -3.0, 2.0), (Normal, 0.0, 1.0), (Laplace, 3.0, 2.0),
        (Logistic, 3.0, 2.0), (Gumbel, 3.0, 2.0), (Gumbel, 500.0, 2500.0))

    for (D, m, s) in cases
        d = reparameterise(D; mean = m, sd = s)
        @test mean(d)≈m atol=1e-8 rtol=1e-8
        @test std(d)≈s rtol=1e-8
    end
end

@testitem "the fallback covers a wide range of coefficients of variation" begin
    using Distributions

    for m in (0.5, 1.0, 8.0, 1.0e4), cv in (0.05, 0.2, 1.0, 3.0)

        d = reparameterise(Frechet; mean = m, sd = m * cv)
        @test mean(d)≈m rtol=1e-8
        @test std(d)≈m * cv rtol=1e-8
    end
end

@testitem "the fallback answers (mean, var) at sqrt(var)" begin
    using Distributions

    by_sd = reparameterise(Frechet; mean = 8.0, sd = 3.0)
    by_var = reparameterise(Frechet; mean = 8.0, var = 9.0)

    @test native(by_var) ≈ native(by_sd)
    @test params(by_var) == (8.0, 9.0)
    @test_throws DomainError reparameterise(Frechet; mean = 8.0, var = -9.0)
end

@testitem "the fallback answers a one-parameter family from (mean,)" begin
    using Distributions

    for (D, m) in ((Rayleigh, 3.0), (Chisq, 4.0), (Exponential, 5.0))
        d = reparameterise(D; mean = m)
        @test mean(d)≈m rtol=1e-8
    end

    # A discrete one, which also checks the wrapper keeps the family's
    # value support through the generic path.
    p = reparameterise(Poisson; mean = 4.0)
    @test native(p) ≈ Poisson(4.0)
    @test p isa DiscreteUnivariateDistribution
    @test logpdf(p, 3) ≈ logpdf(Poisson(4.0), 3)
end

@testitem "the fallback answers a discrete family from (mean, sd)" begin
    using Distributions

    # `NegativeBinomial` registers two closed forms, both in dispersion
    # coordinates, so `(mean, sd)` reaches the fallback. Its success
    # probability is a `:unit` parameter (src/families.jl).
    d = reparameterise(NegativeBinomial; mean = 10.0, sd = 5.0)

    @test d isa DiscreteUnivariateDistribution
    @test mean(d)≈10.0 rtol=1e-8
    @test std(d)≈5.0 rtol=1e-8
    # Against the same distribution reached through a registered closed
    # form: var = mean + mean^2 / dispersion gives dispersion = 20/3.
    @test native(d) ≈ native(reparameterise(NegativeBinomial; mean = 10.0,
        dispersion = 10.0^2 / (5.0^2 - 10.0)))
end

# --- A registered closed form must keep winning ---------------------------

@testitem "every registered closed form still beats the fallback" begin
    using Distributions
    import ReparameterisedDistributions as RD

    # The generic methods are registered under a type variable, so a
    # family's own method is strictly more specific and dispatch picks it.
    # `Frechet` has no closed form, so whichever method it lands on IS the
    # generic one, and no registered family may land on the same method.
    args(D) = (Type{D}, Val{(:mean, :sd)}, Tuple{Float64, Float64})
    generic = which(RD.to_native, args(Frechet))

    for D in (LogNormal, Gamma, Beta, InverseGaussian, Weibull)
        @test which(RD.to_native, args(D)) !== generic
        @test which(RD.valid_moments, args(D)) !==
              which(RD.valid_moments, args(Frechet))
    end

    # And the closed forms still return what the algebra says, to the last
    # bit — a root-find would agree only to its own tolerance.
    m, s = 8.0, 3.0
    @test native(reparameterise(Gamma; mean = m, sd = s)) ==
          Gamma(m^2 / s^2, s^2 / m)
    s2 = log1p((s / m)^2)
    @test native(reparameterise(LogNormal; mean = m, sd = s)) ==
          LogNormal(log(m) - s2 / 2, sqrt(s2))
    @test native(reparameterise(InverseGaussian; mean = m, sd = s)) ==
          InverseGaussian(m, m^3 / s^2)
end

@testitem "a registered (mean, sd) supplies (mean, var) through the fallback" begin
    using Distributions
    import ReparameterisedDistributions: to_native, valid_moments

    # `SymTriangularDist(mu, sigma)` has mean `mu` and variance
    # `sigma^2 / 6`, and this package does not register it. Registering
    # only `(:mean, :sd)`, as a downstream package might, is enough:
    # the generic `(:mean, :var)` delegates to it at `sqrt(var)`.
    function ReparameterisedDistributions.valid_moments(
            ::Type{SymTriangularDist}, ::Val{(:mean, :sd)}, vals)
        _, sd = vals
        return sd > 0
    end

    function ReparameterisedDistributions.to_native(
            ::Type{SymTriangularDist}, ::Val{(:mean, :sd)}, vals)
        m, sd = vals
        return SymTriangularDist(m, sd * sqrt(oftype(sd, 6));
            check_args = false)
    end

    d = reparameterise(SymTriangularDist; mean = 3.0, var = 4.0)
    @test native(d) === SymTriangularDist(3.0, 2.0 * sqrt(6.0))
    @test std(d) ≈ 2.0
end

# --- The arity rule -------------------------------------------------------

@testitem "the fallback needs one moment per native parameter" begin
    using Distributions

    # Under-determined: three native parameters, two moments.
    @test_throws ArgumentError reparameterise(SkewNormal; mean = 1.0,
        sd = 2.0)
    try
        reparameterise(SkewNormal; mean = 1.0, sd = 2.0)
    catch e
        msg = sprint(showerror, e)
        @test occursin("SkewNormal", msg)
        @test occursin("3 native parameters", msg)
        @test occursin("2 standard moments", msg)
    end

    # Over-determined the other way: two native parameters, one moment.
    @test_throws ArgumentError reparameterise(Frechet; mean = 8.0)

    # `check_args = false` skips the construction check entirely, here as
    # for an unregistered pair, so the arity error surfaces on first use
    # instead. It is not silenced: an arity mismatch is fixed by the
    # wrapper's own type parameters, so it throws on every call rather
    # than yielding the `-Inf` an out-of-range VALUE gets.
    d = reparameterise(Frechet; mean = 8.0, check_args = false)
    @test_throws ArgumentError logpdf(d, 7.0)
    @test_throws ArgumentError native(d)
end

# --- The guard, and the -Inf contract -------------------------------------

@testitem "the fallback's guard answers honestly" begin
    using Distributions
    import ReparameterisedDistributions as RD

    @test RD.valid_moments(Frechet, Val((:mean, :sd)), (8.0, 3.0))
    # A negative mean has no Frechet, which has positive support.
    @test !RD.valid_moments(Frechet, Val((:mean, :sd)), (-8.0, 3.0))
    @test !RD.valid_moments(Frechet, Val((:mean, :sd)), (8.0, -3.0))
    @test !RD.valid_moments(Frechet, Val((:mean, :sd)), (Inf, 3.0))
    # A NegativeBinomial is overdispersed relative to a Poisson, so a
    # variance below the mean describes no member of the family.
    @test !RD.valid_moments(NegativeBinomial, Val((:mean, :sd)), (10.0, 2.0))

    @test_throws DomainError reparameterise(NegativeBinomial; mean = 10.0,
        sd = 2.0)
    @test_throws DomainError reparameterise(Frechet; mean = -8.0, sd = 3.0)
end

@testitem "an unsolvable request gives -Inf, not a throw" begin
    using Distributions

    d = reparameterise(NegativeBinomial; mean = 10.0, sd = 2.0,
        check_args = false)

    @test logpdf(d, 3) == -Inf
    @test pdf(d, 3) == 0.0
    @test loglikelihood(d, [3, 4]) == -Inf

    # Every other method converts, so it raises rather than answering.
    @test_throws DomainError mean(d)
    @test_throws DomainError native(d)

    # The `_restype` contract holds through the generic path too.
    f32 = reparameterise(Frechet; mean = -8.0f0, sd = 3.0f0,
        check_args = false)
    @test logpdf(f32, 5.0f0) === -Inf32
    @test pdf(f32, 5.0f0) === 0.0f0
end

# --- Types, inference and cost --------------------------------------------

@testitem "the fallback is type-inferred and allocation-free" begin
    using Distributions, Test
    import ReparameterisedDistributions as RD

    @noinline build(m::Float64, s::Float64) = reparameterise(
        Frechet; mean = m, sd = s)
    @inferred build(8.0, 3.0)
    d = build(8.0, 3.0)
    @inferred native(d)
    @inferred logpdf(d, 7.0)
    @inferred mean(d)

    # The conversion is inferred to a single concrete family member, and
    # never admits `nothing` — the #86 invariant, on the generic path.
    rts = Base.return_types(RD.to_native,
        (Type{Frechet}, Val{(:mean, :sd)}, Tuple{Float64, Float64}))
    @test length(rts) == 1
    @test only(rts) === Frechet{Float64}
    @test !(Nothing <: only(rts))

    native(d)
    logpdf(d, 7.0)
    @test @allocated(native(d)) == 0
    @test @allocated(logpdf(d, 7.0)) == 0
end

@testitem "the fallback keeps a Float32 wrapper in Float32" begin
    using Distributions, Test

    @inferred reparameterise(Frechet; mean = 8.0f0, sd = 3.0f0)
    d = reparameterise(Frechet; mean = 8.0f0, sd = 3.0f0)

    @test native(d) isa Frechet{Float32}
    @test mean(d)≈8.0f0 rtol=1e-4
    @test std(d)≈3.0f0 rtol=1e-4
end

@testitem "the differentiated call sites bind no Nothing union" begin
    using Distributions, Test

    # The same scan as test/families.jl runs over the registered families,
    # on the generic path this time. `optimize = false` is load-bearing:
    # optimised IR union-splits and would miss exactly this.
    function _nothing_unions(f, tt)
        ci, = only(code_typed(f, tt; optimize = false))
        ts = vcat(collect(something(ci.slottypes, [])),
            collect(ci.ssavaluetypes))
        return filter(t -> t isa Union && Nothing <: t, ts)
    end

    for d in (reparameterise(Frechet; mean = 8.0, sd = 3.0),
        reparameterise(Normal; mean = 8.0, sd = 3.0),
        reparameterise(Poisson; mean = 4.0))
        @test isempty(_nothing_unions(logpdf, (typeof(d), Float64)))
        @test isempty(_nothing_unions(pdf, (typeof(d), Float64)))
        @test isempty(_nothing_unions(loglikelihood,
            (typeof(d), Vector{Float64})))
        @test isempty(_nothing_unions(native, (typeof(d),)))
    end
end

@testitem "the fallback passes the registration interface suite" begin
    using Distributions, Test
    using ReparameterisedDistributions: test_reparameterisation

    # The suite an external registration is held to, run against a family
    # nobody registered: the fallback owes a caller exactly what a
    # registration owes one.
    test_reparameterisation(Frechet, (:mean, :sd), (8.0, 3.0);
        invalid = ((8.0, -3.0), (-8.0, 3.0)))
    test_reparameterisation(Normal, (:mean, :sd), (-3.0, 2.0);
        invalid = ((-3.0, -2.0),))
    test_reparameterisation(Rayleigh, (:mean,), (3.0,); invalid = ((-3.0,),))
end

@testitem "the fallback rescales through the same solve" begin
    using Distributions

    d = reparameterise(Frechet; mean = 8.0, sd = 3.0)
    scaled = rescale(d, 2.0)

    @test mean(scaled)≈16.0 rtol=1e-8
    # `rescale` holds every other registered parameter fixed, `sd` here.
    @test std(scaled)≈3.0 rtol=1e-8
    @test params(scaled)[2] == params(d)[2]
end

@testitem "the fallback prints as any other wrapper does" begin
    using Distributions

    d = reparameterise(Frechet; mean = 8.0, sd = 3.0)
    @test occursin("Frechet", sprint(show, "text/plain", d))
    @test occursin("mean = 8.0", sprint(show, d))

    bad = reparameterise(NegativeBinomial; mean = 10.0, sd = 2.0,
        check_args = false)
    @test occursin("invalid parameters", sprint(show, "text/plain", bad))
end

# --- native_domains -------------------------------------------------------

@testitem "native_domains defaults to positive and takes the count from the fields" begin
    using Distributions
    using ReparameterisedDistributions: native_domains

    @test native_domains(Frechet) == (:positive, :positive)
    @test native_domains(Rayleigh) == (:positive,)
    @test native_domains(SkewNormal) ==
          (:positive, :positive, :positive)

    # The families this package registers domains for.
    @test native_domains(Normal) == (:real, :positive)
    @test native_domains(Laplace) == (:real, :positive)
    @test native_domains(Logistic) == (:real, :positive)
    @test native_domains(Gumbel) == (:real, :positive)
    @test native_domains(NegativeBinomial) == (:positive, :unit)
end

@testitem "a family registers native_domains to fix its own coordinates" begin
    using Distributions
    using ReparameterisedDistributions: native_domains

    # `Levy(mu, sigma)` has an unconstrained location, so the default
    # (every parameter positive) cannot reach a negative one. This is the
    # one-line registration a family needs, and nothing else.
    ReparameterisedDistributions.native_domains(::Type{Levy}) = (:real, :positive)

    @test native_domains(Levy) == (:real, :positive)
    # A Levy has neither a finite mean nor a finite variance, so the solve
    # itself still cannot answer `(mean, sd)` for it: the domains say where
    # to look, not that a solution exists.
    @test !ReparameterisedDistributions.valid_moments(
        Levy, Val((:mean, :sd)), (3.0, 2.0))
end

# --- The driver itself ----------------------------------------------------

@testitem "solve_moments inverts a system in the caller's own type" begin
    using Distributions, ForwardDiff
    using ReparameterisedDistributions: solve_moments

    # A system whose solution is known: exp(z) = vals.
    residual = (z, vals) -> (exp(z[1]) - vals[1], exp(z[2]) - vals[2])
    seeds = pvals -> ((zero(pvals[1]), zero(pvals[2])),)

    z = solve_moments(Gamma, Val((:a, :b)), residual, seeds, (2.0, 5.0))
    @test z[1]≈log(2.0) rtol=1e-12
    @test z[2]≈log(5.0) rtol=1e-12

    # The derivative comes from the correction, not from the iteration,
    # so it is exact: d(log v)/dv = 1 / v.
    j = ForwardDiff.jacobian(
        v -> collect(solve_moments(Gamma, Val((:a, :b)), residual, seeds,
            (v[1], v[2]))), [2.0, 5.0])
    @test j ≈ [1/2.0 0.0; 0.0 1/5.0]
end

@testitem "solve_moments reports a system it cannot solve" begin
    using Distributions
    using ReparameterisedDistributions: solve_moments

    # `exp(z) + 1` never reaches zero.
    residual = (z, vals) -> (exp(z[1]) + one(vals[1]),)
    seeds = pvals -> ((zero(pvals[1]),),)

    @test_throws DomainError solve_moments(Gamma, Val((:a,)), residual,
        seeds, (1.0,))
    try
        solve_moments(Gamma, Val((:a,)), residual, seeds, (1.0,))
    catch e
        msg = sprint(showerror, e)
        @test occursin("Gamma", msg)
        @test occursin("did not converge", msg)
        @test occursin("native_domains", msg)
    end
end

@testitem "solve_moments handles one and two unknowns only" begin
    using Distributions
    import ReparameterisedDistributions as RD

    jac = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    @test_throws ArgumentError RD._linear_solve(jac, (1.0, 1.0, 1.0))
end

# --- Automatic differentiation --------------------------------------------

@testitem "the fallback's gradient matches an exact closed form" begin
    using Distributions, ForwardDiff

    # `Normal(mean, sd)` is not registered, so this goes through the
    # solve — and its answer is known exactly, which makes the native
    # distribution an oracle for both the value and the gradient.
    obs = [4.2, 7.1, 9.8, 12.4, 6.0]
    solved(θ) = sum(
        x -> logpdf(
            reparameterise(Normal; mean = θ[1], sd = θ[2],
                check_args = false), x), obs)
    exact(θ) = sum(x -> logpdf(Normal(θ[1], θ[2]), x), obs)

    for θ in ([8.0, 2.0], [-3.0, 0.5], [100.0, 40.0])
        @test solved(θ) ≈ exact(θ)
        @test ForwardDiff.gradient(solved, θ) ≈
              ForwardDiff.gradient(exact, θ)
        @test ForwardDiff.hessian(solved, θ)≈ForwardDiff.hessian(exact, θ) rtol=1e-8
    end
end

@testitem "the fallback's gradient matches central differences" begin
    using Distributions, ForwardDiff

    obs = [4.2, 7.1, 9.8, 12.4, 6.0]
    function loglik(θ)
        d = reparameterise(Frechet; mean = θ[1], sd = θ[2],
            check_args = false)
        return sum(x -> logpdf(d, x), obs)
    end

    for cv in (0.1, 0.4, 1.0, 2.0)
        θ = [8.0, 8.0 * cv]
        g = ForwardDiff.gradient(loglik, θ)

        h = 1e-6
        cd = [(loglik([θ[1] + h, θ[2]]) - loglik([θ[1] - h, θ[2]])) / 2h,
            (loglik([θ[1], θ[2] + h]) - loglik([θ[1], θ[2] - h])) / 2h]
        @test g≈cd rtol=1e-6
        # The truncation trap the correction exists to avoid: a solve that
        # dropped its Duals returns exactly zero here.
        @test all(!=(0), g)
    end
end

@testitem "the fallback's Hessian matches a central difference of the gradient" begin
    using Distributions, ForwardDiff

    obs = [4.2, 7.1, 9.8, 12.4, 6.0]
    function loglik(θ)
        d = reparameterise(Frechet; mean = θ[1], sd = θ[2],
            check_args = false)
        return sum(x -> logpdf(d, x), obs)
    end

    θ = [8.0, 3.0]
    H = ForwardDiff.hessian(loglik, θ)

    h = 1e-6
    g(v) = ForwardDiff.gradient(loglik, v)
    cd_H = reshape(
        [(g([θ[1] + h, θ[2]])[1] - g([θ[1] - h, θ[2]])[1]) / 2h,
            (g([θ[1] + h, θ[2]])[2] - g([θ[1] - h, θ[2]])[2]) / 2h,
            (g([θ[1], θ[2] + h])[1] - g([θ[1], θ[2] - h])[1]) / 2h,
            (g([θ[1], θ[2] + h])[2] - g([θ[1], θ[2] - h])[2]) / 2h], 2, 2)

    @test H≈cd_H rtol=1e-6
end

@testitem "a Dual survives the fallback's solve" begin
    using Distributions, ForwardDiff

    d = reparameterise(Frechet;
        mean = ForwardDiff.Dual(8.0, 1.0, 0.0),
        sd = ForwardDiff.Dual(3.0, 0.0, 1.0), check_args = false)
    nd = native(d)

    @test nd isa Frechet{<:ForwardDiff.Dual}
    for p in (Distributions.shape(nd), Distributions.scale(nd))
        @test all(isfinite, ForwardDiff.partials(p))
        @test all(!=(0), ForwardDiff.partials(p))
    end
end
