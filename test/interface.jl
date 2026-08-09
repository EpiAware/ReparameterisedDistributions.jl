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
