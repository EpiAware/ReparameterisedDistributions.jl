# The point of the wrapper: the moments are the estimable parameters, so a model
# can put priors on them directly and a sampler moves in moment coordinates. A
# reparameterised distribution has to work on the right of a `~` for that, which
# is what these check. Tagged so they can be skipped during fast local
# iteration — Turing is a heavy load.

@testitem "Turing: a reparameterised distribution works on the right of ~" tags=[:turing] begin
    using Distributions, Turing, Random

    @model function delays(x)
        m ~ LogNormal(2.0, 0.5)
        s ~ truncated(Normal(2.0, 1.0); lower = 0.1)
        for i in eachindex(x)
            x[i] ~ reparameterise(LogNormal; mean = m, sd = s, check_args = false)
        end
    end

    obs = [4.2, 7.1, 9.8, 12.4, 6.0]
    model = delays(obs)

    # The `~` scores exactly as the moments imply, with the priors on top: the
    # wrapper is transparent to the likelihood.
    m, s = 8.0, 2.0
    nd = native(
        reparameterise(LogNormal; mean = m, sd = s))
    expected = logpdf(LogNormal(2.0, 0.5), m) +
               logpdf(truncated(Normal(2.0, 1.0); lower = 0.1), s) +
               sum(x -> logpdf(nd, x), obs)
    @test logjoint(model, (m = m, s = s)) ≈ expected
end

@testitem "Turing: the moments are what gets sampled" tags=[:turing] begin
    using Distributions, Turing, Random

    @model function delays(x)
        m ~ LogNormal(2.0, 0.5)
        s ~ truncated(Normal(2.0, 1.0); lower = 0.1)
        for i in eachindex(x)
            x[i] ~ reparameterise(LogNormal; mean = m, sd = s, check_args = false)
        end
    end

    obs = rand(Xoshiro(1),
        native(
            reparameterise(LogNormal; mean = 8.0, sd = 2.0)), 200)

    chain = sample(Xoshiro(2), delays(obs), NUTS(), 200; progress = false)

    # The chain is in MOMENT coordinates — `m` and `s`, not a native `mu`/`sigma`
    # that only implies them. That is the whole point of the package.
    @test :m in names(chain, :parameters)
    @test :s in names(chain, :parameters)
    @test :mu ∉ names(chain, :parameters)
    @test :sigma ∉ names(chain, :parameters)

    # And the sampler, moving in those coordinates, recovers the moments the
    # data were generated with. This only works if the gradient with respect to
    # the moments is right, so it exercises the closed form under AD end to end.
    @test mean(chain[:m])≈8.0 rtol=0.15
    @test mean(chain[:s])≈2.0 rtol=0.3
end
