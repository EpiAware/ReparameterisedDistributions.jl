@testitem "Gamma(mean, sd): exact closed form" begin
    using Distributions

    m, s = 8.0, 3.0
    d = reparameterise(Gamma; mean = m, sd = s)

    # A Gamma(shape, scale) has mean = shape * scale and var = shape * scale^2,
    # so scale = var / mean and shape = mean^2 / var.
    scale = s^2 / m
    shape = m^2 / s^2
    @test native(d) ≈ Gamma(shape, scale)

    # The conversion is exact, so the moments come back out.
    @test params(d) == (m, s)
    @test mean(d) ≈ m
    @test std(d) ≈ s
    @test var(d) ≈ s^2
end

@testitem "Gamma(mean, var) agrees with Gamma(mean, sd)" begin
    using Distributions

    by_sd = reparameterise(Gamma; mean = 8.0, sd = 3.0)
    by_var = reparameterise(Gamma; mean = 8.0, var = 9.0)

    @test native(by_var) ≈
          native(by_sd)
    @test params(by_var) == (8.0, 9.0)
end

@testitem "Gamma(mean, shape): only the scale is derived" begin
    using Distributions

    m, shape = 8.0, 3.0
    d = reparameterise(Gamma; mean = m, shape = shape)

    # The shape is native; the scale is mean / shape. This is the pair
    # CensoredDistributions registered.
    @test native(d) ≈ Gamma(shape, m / shape)
    @test params(d) == (m, shape)
    @test mean(d) ≈ m
    # The implied standard deviation follows from the shape.
    @test std(d) ≈ m / sqrt(shape)
end

@testitem "Exponential(rate): the native scale inverted" begin
    using Distributions

    rate = 0.5
    d = reparameterise(Exponential; rate = rate)

    # The native Exponential(θ) takes the scale, θ = 1 / rate.
    @test native(d) ≈ Exponential(1 / rate)
    @test params(d) == (rate,)
    @test mean(d) ≈ 1 / rate
end

@testitem "Gamma(rate, shape): the reciprocal of (mean, shape)" begin
    using Distributions

    rate, shape = 0.5, 2.0
    d = reparameterise(Gamma; shape = shape, rate = rate)

    # The shape is native; the scale is 1 / rate.
    @test native(d) ≈ Gamma(shape, 1 / rate)
    # The names sort alphabetically ('r' < 's'), so `params` reports
    # (rate, shape) regardless of the keyword order at the call site.
    @test params(d) == (rate, shape)
    @test mean(d) ≈ shape / rate
end

@testitem "SkewNormal(centre, scale, mass_below_centre): the tail-probability form" begin
    using Distributions

    centre, scale, m = 0.0, 1.0, 0.3
    d = reparameterise(SkewNormal; centre = centre, scale = scale,
        mass_below_centre = m)

    # alpha = tan(pi * (1/2 - m)) is the exact inverse of
    # P(X < centre) = 1/2 - atan(alpha) / pi for the untruncated family.
    alpha = tan(pi * (1 / 2 - m))
    @test native(d) ≈ SkewNormal(centre, scale, alpha)

    # Names sort alphabetically ('c' < 'm' < 's'), so `params` reports
    # (centre, mass_below_centre, scale) regardless of call-site order.
    @test params(d) == (centre, m, scale)

    # Distributions.jl has no `cdf` for SkewNormal (Owen's T is not
    # implemented there), so the elicited mass is checked against the closed
    # form directly rather than via `cdf(d, centre)`, which would MethodError
    # for a native SkewNormal too.
    @test 1 / 2 - atan(alpha) / pi≈m rtol=1e-12
end

@testitem "SkewNormal: a symmetric (m = 1/2) elicitation gives alpha = 0" begin
    using Distributions

    # Half the mass below the centre is the symmetric case: no skew.
    d = reparameterise(SkewNormal; centre = 1.0, scale = 2.0,
        mass_below_centre = 0.5)
    _, _, alpha = params(native(d))
    @test alpha≈0.0 atol=1e-12
    @test mean(d) ≈ 1.0
end

@testitem "SkewNormal: the closed form validates its moments" begin
    using Distributions

    @test_throws DomainError reparameterise(SkewNormal; centre = 0.0,
        scale = -1.0, mass_below_centre = 0.3)
    @test_throws DomainError reparameterise(SkewNormal; centre = 0.0,
        scale = 1.0, mass_below_centre = 0.0)
    @test_throws DomainError reparameterise(SkewNormal; centre = 0.0,
        scale = 1.0, mass_below_centre = 1.0)
end

@testitem "SkewNormal is usable through the density/sampling Distributions interface" begin
    using Distributions, Random, Statistics

    d = reparameterise(SkewNormal; centre = 0.0, scale = 1.0,
        mass_below_centre = 0.3)
    nd = native(d)

    # cdf/quantile are not exercised here: Distributions.jl does not
    # implement them for SkewNormal at all, native or wrapped.
    @test logpdf(d, 0.5) ≈ logpdf(nd, 0.5)
    @test pdf(d, 0.5) ≈ pdf(nd, 0.5)
    @test mean(d) ≈ mean(nd)
    @test var(d) ≈ var(nd)

    draws = rand(Xoshiro(1), d, 20_000)
    @test Statistics.mean(draws)≈mean(d) rtol=0.05
end

@testitem "NegativeBinomial(mean, overdispersion): the epi parameterisation" begin
    using Distributions

    m, a = 10.0, 0.1
    d = reparameterise(NegativeBinomial; mean = m, overdispersion = a)

    # The defining relation is var = mean + a * mean^2, so a is the excess
    # variance relative to a Poisson.
    @test mean(d) ≈ m
    @test var(d)≈m + a * m^2 rtol=1e-10

    # Against the native parameters worked by hand: r = 1/a, p = 1/(1 + a*mean).
    @test native(d) ≈
          NegativeBinomial(1 / a, 1 / (1 + a * m))

    @test params(d) == (m, a)
end

@testitem "NegativeBinomial(mean, dispersion): the reciprocal convention" begin
    using Distributions

    m, k = 10.0, 2.0
    d = reparameterise(NegativeBinomial; mean = m, dispersion = k)

    # The defining relation is var = mean + mean^2 / dispersion, the reciprocal
    # of the overdispersion convention's var = mean + overdispersion * mean^2.
    @test mean(d) ≈ m
    @test var(d)≈m + m^2 / k rtol=1e-10

    # Against the native parameters worked by hand: r = dispersion,
    # p = dispersion / (dispersion + mean).
    @test native(d) ≈ NegativeBinomial(k, k / (k + m))

    # The two conventions are reciprocals, so equal spread comes from
    # `dispersion = 1 / overdispersion`.
    a = 1 / k
    @test native(d) ≈
          native(reparameterise(NegativeBinomial; mean = m, overdispersion = a))

    # The names sort alphabetically ('d' < 'm'), so `params` reports
    # (dispersion, mean) regardless of the keyword order at the call site.
    @test params(d) == (k, m)
    @test params(reparameterise(NegativeBinomial; dispersion = k, mean = m)) ==
          (k, m)
end

@testitem "NegativeBinomial: larger dispersion approaches the Poisson" begin
    using Distributions

    m = 10.0
    # As dispersion -> Inf the variance falls to the mean, the Poisson limit —
    # the opposite direction from the overdispersion convention, where the
    # limit is `a -> 0`.
    @test var(reparameterise(NegativeBinomial; mean = m,
        dispersion = 1e6))≈m rtol=1e-4
    @test var(reparameterise(NegativeBinomial; mean = m,
        dispersion = 2.0)) > m

    @test_throws DomainError reparameterise(NegativeBinomial; mean = m,
        dispersion = 0.0)
    @test_throws DomainError reparameterise(NegativeBinomial; mean = m,
        dispersion = -1.0)
end

@testitem "NegativeBinomial stays DISCRETE" begin
    using Distributions

    d = reparameterise(NegativeBinomial; mean = 10.0, overdispersion = 0.1)

    # The wrapper takes its value support from the family it wraps, so a
    # discrete family does not silently become continuous.
    @test d isa DiscreteUnivariateDistribution
    @test Distributions.value_support(typeof(d)) == Discrete
    @test insupport(d, 3)
    @test !insupport(d, 3.5)

    # And a continuous family stays continuous.
    c = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    @test c isa ContinuousUnivariateDistribution
end

@testitem "NegativeBinomial: smaller overdispersion approaches the Poisson" begin
    using Distributions

    m = 10.0
    # As a -> 0 the variance falls to the mean, which is the Poisson limit.
    @test var(reparameterise(NegativeBinomial; mean = m,
        overdispersion = 1e-6))≈m rtol=1e-4
    @test var(reparameterise(NegativeBinomial; mean = m,
        overdispersion = 0.5)) > m

    # The limit itself is not a NegativeBinomial: r = 1 / a diverges.
    @test_throws DomainError reparameterise(NegativeBinomial; mean = m,
        overdispersion = 0.0)
end

@testitem "the closed forms validate their moments" begin
    using Distributions

    @test_throws DomainError reparameterise(Gamma; mean = 8.0, sd = -1.0)
    @test_throws DomainError reparameterise(Gamma; mean = -8.0, sd = 1.0)
    @test_throws DomainError reparameterise(Gamma; mean = 8.0, shape = -1.0)
    @test_throws DomainError reparameterise(Gamma; mean = 8.0, var = -1.0)
    @test_throws DomainError reparameterise(NegativeBinomial; mean = -1.0,
        overdispersion = 0.1)
    @test_throws DomainError reparameterise(NegativeBinomial; mean = 10.0,
        overdispersion = -0.1)
    @test_throws DomainError reparameterise(NegativeBinomial; mean = -1.0,
        dispersion = 2.0)
    @test_throws DomainError reparameterise(NegativeBinomial; mean = 10.0,
        dispersion = -2.0)
    @test_throws DomainError reparameterise(Exponential; rate = -0.5)
    @test_throws DomainError reparameterise(Exponential; rate = 0.0)
    @test_throws DomainError reparameterise(Gamma; shape = 2.0, rate = -0.5)
    @test_throws DomainError reparameterise(Gamma; shape = -2.0, rate = 0.5)
end

@testitem "the closed forms are usable through the Distributions interface" begin
    using Distributions, Random, Statistics

    for d in (reparameterise(Gamma; mean = 8.0, sd = 3.0),
        reparameterise(Gamma; mean = 8.0, shape = 3.0),
        reparameterise(NegativeBinomial; mean = 10.0, overdispersion = 0.1),
        reparameterise(NegativeBinomial; mean = 10.0, dispersion = 5.0),
        reparameterise(Exponential; rate = 0.5),
        reparameterise(Gamma; shape = 3.0, rate = 0.5))
        nd = native(d)
        x = minimum(d) == 0 ? 4 : 4.0

        @test logpdf(d, x) ≈ logpdf(nd, x)
        @test cdf(d, x) ≈ cdf(nd, x)
        @test quantile(d, 0.4) ≈ quantile(nd, 0.4)
        @test mean(d) ≈ mean(nd)
        @test var(d) ≈ var(nd)

        # And it draws with the moments it is named by.
        draws = rand(Xoshiro(1), d, 20_000)
        @test Statistics.mean(draws)≈mean(d) rtol=0.05
    end
end
