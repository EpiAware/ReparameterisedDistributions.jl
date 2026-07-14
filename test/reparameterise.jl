@testitem "reparameterise: the moments are the parameters" begin
    using Distributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)

    # The whole point: `params` reports the moments, not the native (mu, sigma).
    @test params(d) == (8.0, 2.0)
    @test params(d) != params(ReparameterisedDistributions._native(d))

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
    @test ReparameterisedDistributions._native(d) ≈ expected

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
    native = ReparameterisedDistributions._native(d)
    x = 7.5

    @test logpdf(d, x) ≈ logpdf(native, x)
    @test pdf(d, x) ≈ pdf(native, x)
    @test cdf(d, x) ≈ cdf(native, x)
    @test logcdf(d, x) ≈ logcdf(native, x)
    @test ccdf(d, x) ≈ ccdf(native, x)
    @test logccdf(d, x) ≈ logccdf(native, x)
    @test quantile(d, 0.4) ≈ quantile(native, 0.4)
    @test insupport(d, x) == insupport(native, x)
    @test minimum(d) == minimum(native)
    @test maximum(d) == maximum(native)
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

    # That the native distribution really would have been valid is the point:
    # the guard cannot be delegated to Distributions.jl.
    bad = reparameterise(LogNormal; mean = 8.0, sd = -1.0, check_args = false)
    good = reparameterise(LogNormal; mean = 8.0, sd = 1.0, check_args = false)
    @test ReparameterisedDistributions._native(bad) ≈
          ReparameterisedDistributions._native(good)

    # With the check off, construction succeeds — a sampler probing an invalid
    # point needs a density, not an exception, mid-gradient.
    @test bad isa Distribution
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
