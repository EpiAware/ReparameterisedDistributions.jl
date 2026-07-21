@testitem "rescale: scales the default :mean, holds the rest fixed" begin
    using Distributions

    d = reparameterise(Gamma; mean = 8.0, shape = 2.0)
    scaled = rescale(d, 2.0)

    @test mean(scaled) ≈ 16.0
    @test params(scaled) == (16.0, 2.0)
    # The shape — the other registered parameter — is untouched.
    @test params(scaled)[2] == params(d)[2]
end

@testitem "rescale: routes through the discrete family's own conversion" begin
    using Distributions

    # An affine transform is not a substitute here: the wrapper has to scale
    # the mean in moment coordinates and reconvert, not scale the native
    # support directly.
    nb = reparameterise(NegativeBinomial; mean = 10.0, overdispersion = 0.5)
    scaled = rescale(nb, 3.0)

    @test mean(scaled) ≈ 30.0
    @test scaled isa DiscreteUnivariateDistribution
    @test params(scaled)[2] == params(nb)[2]
end

@testitem "rescale: the parameter keyword selects a different registered name" begin
    using Distributions

    d = reparameterise(Gamma; mean = 8.0, shape = 2.0)
    scaled = rescale(d, 2.0; parameter = :shape)

    @test params(scaled)[1] == params(d)[1]
    @test params(scaled)[2] ≈ 4.0
end

@testitem "rescale: an unsupported parameter throws DomainError" begin
    using Distributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    @test_throws DomainError rescale(d, 2.0; parameter = :variance)
end

@testitem "rescale: a native (unwrapped) distribution throws ArgumentError" begin
    using Distributions

    # There is no registered parameterisation to route through until the
    # family is wrapped by `reparameterise`.
    @test_throws ArgumentError rescale(Gamma(2.0, 4.0), 2.0)
end

@testitem "rescale: check_args validates the scaled moments" begin
    using Distributions

    d = reparameterise(Gamma; mean = 8.0, shape = 2.0)
    @test_throws DomainError rescale(d, -1.0)

    bad = rescale(d, -1.0; check_args = false)
    @test logpdf(bad, 4.0) == -Inf
end

@testitem "rescale: construction is fully inferred" begin
    using Distributions, Test

    d = reparameterise(Gamma; mean = 8.0, shape = 2.0)
    @noinline build(dist, f::Float64) = rescale(dist, f)
    @inferred build(d, 2.0)
end
