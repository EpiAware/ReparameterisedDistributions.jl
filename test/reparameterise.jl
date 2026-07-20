@testitem "reparameterise: the moments are the parameters" begin
    using Distributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)

    # The whole point: `params` reports the moments, not the native (mu, sigma).
    @test params(d) == (8.0, 2.0)
    @test params(d) != params(native(d))

    # And the wrapper is a Distribution, so it can stand where one is expected.
    @test d isa Distribution
    @test d isa ReparameterisedDistributions.AbstractReparameterisedDistribution
end

@testitem "reparameterise: LogNormal(mean, sd) is the exact closed form" begin
    using Distributions

    m, s = 8.0, 2.0
    d = reparameterise(LogNormal; mean = m, sd = s)

    # Against the closed form worked by hand from the log-normal moments.
    s2 = log1p((s / m)^2)
    expected = LogNormal(log(m) - s2 / 2, sqrt(s2))
    @test native(d) ≈ expected

    # The conversion is exact, so the moments come back out.
    @test mean(d) ≈ m
    @test std(d) ≈ s
    @test var(d) ≈ s^2
end

@testitem "reparameterise: var and sd agree" begin
    using Distributions

    by_sd = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    by_var = reparameterise(LogNormal; mean = 8.0, var = 4.0)

    @test mean(by_var) ≈ mean(by_sd)
    @test var(by_var) ≈ var(by_sd)
    # The names differ, so the reported parameters differ: `var` is kept as the
    # parameter when that is how the distribution was specified.
    @test params(by_var) == (8.0, 4.0)
end

@testitem "reparameterise: the full Distributions interface delegates" begin
    using Distributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    nd = native(d)
    x = 7.5

    @test logpdf(d, x) ≈ logpdf(nd, x)
    @test pdf(d, x) ≈ pdf(nd, x)
    @test cdf(d, x) ≈ cdf(nd, x)
    @test logcdf(d, x) ≈ logcdf(nd, x)
    @test ccdf(d, x) ≈ ccdf(nd, x)
    @test logccdf(d, x) ≈ logccdf(nd, x)
    @test quantile(d, 0.4) ≈ quantile(nd, 0.4)
    @test insupport(d, x) == insupport(nd, x)
    @test minimum(d) == minimum(nd)
    @test maximum(d) == maximum(nd)
end

@testitem "reparameterise: rand draws from the native distribution" begin
    using Distributions, Random, Statistics

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    draws = rand(Xoshiro(1), d, 20_000)

    # The moments the distribution is named by are the moments it draws with.
    @test Statistics.mean(draws)≈8.0 rtol=0.05
    @test Statistics.std(draws)≈2.0 rtol=0.1
    @test all(>(0), draws)
end

@testitem "reparameterise: accepts an instance, taking only its family" begin
    using Distributions

    from_type = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    # The instance's own parameter values are irrelevant; only its family is
    # taken.
    from_instance = reparameterise(LogNormal(99.0, 99.0); mean = 8.0, sd = 2.0)

    @test from_instance == from_type
end

@testitem "reparameterise: rejects an unregistered parameterisation" begin
    using Distributions

    @test_throws ArgumentError reparameterise(LogNormal; shape = 2.0, rate = 1.0)
    @test_throws ArgumentError reparameterise(LogNormal)
end

@testitem "reparameterise: check_args validates the moments themselves" begin
    using Distributions

    # The conversion squares `sd / mean`, so a negative standard deviation maps
    # onto exactly the same VALID native LogNormal as its positive counterpart.
    # Checking only the native distribution would therefore let this through,
    # and the wrapper would report an `sd` of -1.0 while behaving as +1.0. The
    # moments have to be checked in their own coordinates.
    @test_throws DomainError reparameterise(LogNormal; mean = 8.0, sd = -1.0)
    @test_throws DomainError reparameterise(LogNormal; mean = -8.0, sd = 2.0)
    @test_throws DomainError reparameterise(LogNormal; mean = 8.0, sd = 0.0)
    @test_throws DomainError reparameterise(LogNormal; mean = 8.0, var = -4.0)
end

@testitem "reparameterise: an invalid point is -Inf, not an error and not a lie" begin
    using Distributions

    # `check_args = false` exists so a sampler exploring an unconstrained
    # parameter gets a density back at an invalid proposal rather than an
    # exception raised mid-gradient. That is a contract, so it is tested.
    for bad in (reparameterise(LogNormal; mean = 8.0, sd = -1.0,
        check_args = false),
        reparameterise(LogNormal; mean = -8.0, sd = 2.0, check_args = false),
        reparameterise(Gamma; mean = 8.0, sd = -1.0, check_args = false),
        reparameterise(Gamma; mean = 8.0, shape = -1.0, check_args = false),
        reparameterise(NegativeBinomial; mean = 10.0, overdispersion = 0.0,
        check_args = false),
        reparameterise(NegativeBinomial; mean = -10.0, overdispersion = 0.1,
        check_args = false))
        @test logpdf(bad, 4.0) == -Inf
        @test pdf(bad, 4.0) == 0.0
    end

    # And the density must not merely be finite-but-wrong. A negative standard
    # deviation converts to a native distribution that is not just valid but
    # IDENTICAL to the one a positive standard deviation gives — so without the
    # guard the density would be finite, equal to the density at +sd, and the
    # sign would be unidentifiable. Pin that the native really does alias, and
    # that the wrapper nonetheless refuses it.
    bad = reparameterise(LogNormal; mean = 8.0, sd = -1.0, check_args = false)
    good = reparameterise(LogNormal; mean = 8.0, sd = 1.0, check_args = false)
    @test native(bad) ≈
          native(good)
    @test logpdf(bad, 7.5) == -Inf
    @test isfinite(logpdf(good, 7.5))
end

@testitem "reparameterise: keyword order does not change the meaning" begin
    using Distributions

    # Julia keywords are order-insensitive everywhere else, and the package's
    # only public entry point must not be the exception.
    @test reparameterise(LogNormal; sd = 2.0, mean = 8.0) ==
          reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    @test reparameterise(Gamma; shape = 3.0, mean = 8.0) ==
          reparameterise(Gamma; mean = 8.0, shape = 3.0)
    @test reparameterise(NegativeBinomial; overdispersion = 0.1, mean = 10.0) ==
          reparameterise(NegativeBinomial; mean = 10.0, overdispersion = 0.1)
end

@testitem "reparameterise: the moment summaries are reported" begin
    using Distributions

    # A package sold on moments should be able to report its moments, rather
    # than reaching a Base generic and failing with an opaque `iterate` error.
    d = reparameterise(Gamma; mean = 8.0, sd = 3.0)
    nd = native(d)

    @test mode(d) ≈ mode(nd)
    @test skewness(d) ≈ skewness(nd)
    @test kurtosis(d) ≈ kurtosis(nd)
    @test entropy(d) ≈ entropy(nd)
    @test mgf(d, 0.1) ≈ mgf(nd, 0.1)
    @test median(d) ≈ median(nd)
    @test std(d) ≈ std(nd)
end

@testitem "reparameterise: parameters promote to a common type" begin
    using Distributions

    d = reparameterise(LogNormal; mean = 8, sd = 2.0)
    @test params(d) == (8.0, 2.0)
    @test eltype(params(d)) == Float64
end

@testitem "reparameterise: show round-trips the parameterisation" begin
    using Distributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    @test occursin("mean = 8.0", sprint(show, d))
    @test occursin("sd = 2.0", sprint(show, d))
end

@testitem "reparameterise: MIME text/plain show also reports the native distribution" begin
    using Distributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    out = sprint(show, MIME("text/plain"), d)
    @test occursin("mean = 8.0", out)
    @test occursin("native:", out)
    @test occursin("LogNormal", out)
end

@testitem "reparameterise: the rebuild hook promotes like the front door" begin
    using Distributions

    # `_build` is the ecosystem's leaf-rebuild entry point, so a leaf rebuilt
    # from a mixed Int/Float tuple must not end up with an abstract
    # `NTuple{2, Real}` field — that would be boxed, type-unstable and hostile
    # to a gradient.
    d = ReparameterisedDistributions._build(
        LogNormal, Val((:mean, :sd)), (8, 2.0))
    @test params(d) === (8.0, 2.0)
    @test eltype(params(d)) === Float64
    @test d == reparameterise(LogNormal; mean = 8.0, sd = 2.0)

    # And it canonicalises the names, so the hook cannot smuggle in an order the
    # front door would reject.
    @test ReparameterisedDistributions._build(LogNormal, Val((:sd, :mean)),
        (2.0, 8.0)) == reparameterise(LogNormal; mean = 8.0, sd = 2.0)
end

@testitem "reparameterise: construction is fully inferred" begin
    using Distributions, Test

    # #45: `_build` used to take `names` as a bare tuple and call `Val(names)`
    # on it internally, which cannot be inferred concretely from a runtime
    # argument — `reparameterise`'s own return type came back with free
    # `names`/`N`/`T` type parameters for any call the compiler did not fully
    # constant-fold. `names` now arrives as a `Val` at the API boundary
    # instead, so the whole `Reparameterised{...}` type is inferred, which
    # matters most exactly where constant folding cannot be relied on: inside
    # an AD tape.
    @noinline build(m::Float64, s::Float64) = reparameterise(
        LogNormal; mean = m, sd = s)
    @inferred build(8.0, 2.0)

    d = build(8.0, 2.0)
    @inferred ReparameterisedDistributions._native(d)
    @inferred logpdf(d, 7.5)
    @inferred mean(d)
end
