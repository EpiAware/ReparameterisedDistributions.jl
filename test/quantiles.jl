@testitem "quantiles: two quantiles pin a Normal exactly" begin
    using Distributions

    d = reparameterise(Normal; quantiles = (0.25 => 1.0, 0.75 => 3.0))

    # The elicited quantiles come back out, which is the whole contract.
    @test quantile(d, 0.25)≈1.0 rtol=1e-12
    @test quantile(d, 0.75)≈3.0 rtol=1e-12

    # Against the native parameters worked by hand: the pair is symmetric
    # about 2.0, and the scale is the half-width over the standard normal's
    # upper quartile.
    @test native(d) ≈ Normal(2.0, 1.0 / quantile(Normal(), 0.75))

    # `params` reports the elicited values, not the native (mu, sigma).
    @test params(d) == (1.0, 3.0)
    @test params(d) != params(native(d))
end

@testitem "quantiles: the exact location-scale inversions" begin
    using Distributions

    # Each pair is symmetric about 2.0, so the location is 2.0 in every case
    # and only the scale differs. `z` is the standard member's own quartile.
    cases = ((Normal, Normal(2.0, 1.0 / quantile(Normal(), 0.75))),
        # z(3/4) = log(3), so the scale is 1 / log(3).
        (Logistic, Logistic(2.0, 1.0 / log(3.0))),
        # z(3/4) = tan(pi / 4) = 1.
        (Cauchy, Cauchy(2.0, 1.0)),
        # z(3/4) = -log(2 * (1 - 3/4)) = log(2).
        (Laplace, Laplace(2.0, 1.0 / log(2.0))),
        # z(p) = p, so the width is (3 - 1) / (3/4 - 1/4) = 4.
        (Uniform, Uniform(0.0, 4.0)))

    for (D, expected) in cases
        d = reparameterise(D; quantiles = (0.25 => 1.0, 0.75 => 3.0))
        @test native(d) ≈ expected
        @test quantile(d, 0.25)≈1.0 rtol=1e-12
        @test quantile(d, 0.75)≈3.0 rtol=1e-12
        @test params(d) == (1.0, 3.0)
    end
end

@testitem "quantiles: LogNormal inverts in log space" begin
    using Distributions

    d = reparameterise(LogNormal; quantiles = (0.05 => 1.2, 0.95 => 8.4))

    # Two quantiles of a LogNormal are two quantiles of a Normal on the logs.
    z05, z95 = quantile(Normal(), 0.05), quantile(Normal(), 0.95)
    sigma = (log(8.4) - log(1.2)) / (z95 - z05)
    @test native(d) ≈ LogNormal(log(1.2) - sigma * z05, sigma)

    @test quantile(d, 0.05)≈1.2 rtol=1e-12
    @test quantile(d, 0.95)≈8.4 rtol=1e-12
    @test params(d) == (1.2, 8.4)
end

@testitem "quantiles: one quantile pins an Exponential exactly" begin
    using Distributions

    d = reparameterise(Exponential; quantiles = (0.95 => 3.0,))

    # Q(p) = -theta * log1p(-p), so theta = 3 / log(20).
    @test native(d) ≈ Exponential(3.0 / log(20.0))
    @test quantile(d, 0.95)≈3.0 rtol=1e-12
    @test params(d) == (3.0,)
end

@testitem "quantiles: arbitrary probabilities, not a fixed convention" begin
    using Distributions

    d = reparameterise(Normal; quantiles = (1 / 3 => 2.0, 2 / 3 => 6.0))

    @test quantile(d, 1 / 3)≈2.0 rtol=1e-12
    @test quantile(d, 2 / 3)≈6.0 rtol=1e-12
    @test mean(d) ≈ 4.0
end

@testitem "quantiles: mixed with a moment keyword" begin
    using Distributions

    # A median and an upper tail point, the shape elicitation usually takes.
    d = reparameterise(LogNormal; median = 4.0, quantiles = (0.95 => 12.0,))
    @test median(d)≈4.0 rtol=1e-12
    @test quantile(d, 0.95)≈12.0 rtol=1e-12
    # The names sort with moments before probabilities.
    @test params(d) == (4.0, 12.0)

    # The mean stands in for the median in a symmetric family.
    n = reparameterise(Normal; mean = 8.0, quantiles = (0.95 => 20.0,))
    @test mean(n)≈8.0 rtol=1e-12
    @test quantile(n, 0.95)≈20.0 rtol=1e-12

    # And a scale constraint mixes just as well as a location one.
    s = reparameterise(Normal; sd = 2.0, quantiles = (0.95 => 20.0,))
    @test std(s)≈2.0 rtol=1e-12
    @test quantile(s, 0.95)≈20.0 rtol=1e-12
end

@testitem "quantiles: the keyword order does not change the meaning" begin
    using Distributions

    @test reparameterise(Normal; quantiles = (0.75 => 3.0, 0.25 => 1.0)) ==
          reparameterise(Normal; quantiles = (0.25 => 1.0, 0.75 => 3.0))
    @test reparameterise(LogNormal; quantiles = (0.95 => 12.0,),
        median = 4.0) ==
          reparameterise(LogNormal; median = 4.0, quantiles = (0.95 => 12.0,))
end

@testitem "quantiles: a lone pair is the one-constraint case" begin
    using Distributions

    @test reparameterise(Exponential; quantiles = 0.95 => 3.0) ==
          reparameterise(Exponential; quantiles = (0.95 => 3.0,))
end

@testitem "quantiles: show round-trips the parameterisation" begin
    using Distributions

    d = reparameterise(Normal; quantiles = (0.25 => 1.0, 0.75 => 3.0))
    @test sprint(show, d) ==
          "reparameterise(Normal; quantiles = (0.25 => 1.0, 0.75 => 3.0))"

    # A single constraint keeps the trailing comma, so the printed form is
    # still a tuple when pasted back.
    e = reparameterise(Exponential; quantiles = (0.95 => 3.0,))
    @test sprint(show, e) ==
          "reparameterise(Exponential; quantiles = (0.95 => 3.0,))"

    m = reparameterise(LogNormal; median = 4.0, quantiles = (0.95 => 12.0,))
    @test sprint(show, m) ==
          "reparameterise(LogNormal; median = 4.0, " *
          "quantiles = (0.95 => 12.0,))"
end

@testitem "quantiles: decreasing values describe no distribution" begin
    using Distributions

    # A larger value at the smaller probability is not a hard solve, it is
    # no distribution at all, so the predicate catches it.
    @test_throws DomainError reparameterise(Normal;
        quantiles = (0.25 => 3.0, 0.75 => 1.0))
    @test_throws DomainError reparameterise(Normal;
        quantiles = (0.25 => 1.0, 0.75 => 1.0))

    bad = reparameterise(Normal; quantiles = (0.25 => 3.0, 0.75 => 1.0),
        check_args = false)
    @test logpdf(bad, 2.0) == -Inf
    @test pdf(bad, 2.0) == 0.0

    # A LogNormal is elicited on a positive support, so a non-positive
    # value is refused before the log transform is taken.
    @test_throws DomainError reparameterise(LogNormal;
        quantiles = (0.05 => -1.0, 0.95 => 8.4))
    worse = reparameterise(LogNormal; quantiles = (0.05 => -1.0, 0.95 => 8.4),
        check_args = false)
    @test logpdf(worse, 2.0) == -Inf
end

@testitem "quantiles: the constraint set is checked structurally" begin
    using Distributions

    # A probability has to be one: strictly inside (0, 1), and distinct.
    @test_throws ArgumentError reparameterise(Normal;
        quantiles = (0.0 => 1.0, 0.75 => 3.0))
    @test_throws ArgumentError reparameterise(Normal;
        quantiles = (0.25 => 1.0, 1.0 => 3.0))
    @test_throws ArgumentError reparameterise(Normal;
        quantiles = (0.25 => 1.0, 0.25 => 3.0))

    # Under- and over-determined requests name both counts.
    @test_throws ArgumentError reparameterise(Normal;
        quantiles = (0.75 => 3.0,))
    @test_throws ArgumentError reparameterise(Normal;
        quantiles = (0.25 => 1.0, 0.5 => 2.0, 0.75 => 3.0))
    try
        reparameterise(Normal; quantiles = (0.75 => 3.0,))
    catch e
        msg = sprint(showerror, e)
        @test occursin("Normal", msg)
        @test occursin("2", msg)
        @test occursin("1", msg)
    end

    # A vector puts neither the count nor the probabilities in the type.
    @test_throws ArgumentError reparameterise(Normal;
        quantiles = [0.25 => 1.0, 0.75 => 3.0])
end

@testitem "quantiles: a family with no exact inversion says so" begin
    using Distributions

    # The generic numeric solve for constraint sets is not wired up yet, so
    # a Gamma elicited by quantiles is refused by name rather than by a
    # bare "not registered".
    @test_throws ArgumentError reparameterise(Gamma;
        quantiles = (0.05 => 1.2, 0.95 => 8.4))
    try
        reparameterise(Gamma; quantiles = (0.05 => 1.2, 0.95 => 8.4))
    catch e
        msg = sprint(showerror, e)
        @test occursin("Gamma", msg)
        @test occursin("numeric", msg)
    end

    # A moment the exact inversion cannot express is refused the same way:
    # a LogNormal's mean is not linear in its native parameters.
    @test_throws ArgumentError reparameterise(LogNormal; mean = 8.0,
        quantiles = (0.95 => 20.0,))
end

@testitem "quantiles: construction is fully inferred" begin
    using Distributions, Test

    # The probabilities are type parameters, so they have to reach the
    # wrapper's type from a call whose VALUES are not constants — which is
    # every call inside a model.
    @noinline build(a::Float64, b::Float64) = reparameterise(
        Normal; quantiles = (0.25 => a, 0.75 => b))
    @inferred build(1.0, 3.0)

    d = build(1.0, 3.0)
    @inferred native(d)
    @inferred logpdf(d, 2.0)
    @inferred mean(d)

    @noinline mixed(m::Float64, q::Float64) = reparameterise(
        LogNormal; median = m, quantiles = (0.95 => q))
    @inferred mixed(4.0, 12.0)
end

@testitem "quantiles: every exact inversion is concrete, never a Nothing union" begin
    using Distributions, Test
    import ReparameterisedDistributions as RD

    # The #86 invariant, for the quantile registrations: they are reached
    # through a catch-all method whose `names` is a type variable, so the
    # sweep over `methods(to_native)` in test/families.jl cannot see them
    # and they are checked here at concrete probability tuples instead.
    cases = ((Normal, (0.25, 0.75)), (LogNormal, (0.05, 0.95)),
        (Logistic, (0.25, 0.75)), (Cauchy, (0.25, 0.75)),
        (Uniform, (0.25, 0.75)), (Laplace, (0.25, 0.75)),
        (Exponential, (0.95,)), (LogNormal, (:median, 0.95)),
        (Normal, (:mean, 0.95)), (Normal, (:sd, 0.95)))

    for (D, names) in cases
        args = (Type{D}, Val{names}, NTuple{length(names), Float64})
        rts = Base.return_types(RD.to_native, args)
        @test length(rts) == 1
        rt = only(rts)
        @test !(Nothing <: rt)
        @test rt <: D
        @test isconcretetype(rt)
    end
end

@testitem "quantiles: an unregistered moment pair still raises, not -Inf" begin
    using Distributions
    using ReparameterisedDistributions: valid_moments

    # A family this file catches all names for must not swallow a name set
    # it cannot convert: the predicate stays `true` so the conversion's own
    # error surfaces, rather than the request being silently zero-density.
    @test valid_moments(Normal, Val((:mean, :var)), (8.0, 4.0)) == true
    @test_throws ArgumentError reparameterise(Normal; mean = 8.0, var = 4.0)
end

@testitem "quantiles: the exact inversions are usable through the interface" begin
    using Distributions, Random, Statistics

    for d in (reparameterise(Normal; quantiles = (0.25 => 1.0, 0.75 => 3.0)),
        reparameterise(LogNormal; quantiles = (0.05 => 1.2, 0.95 => 8.4)),
        reparameterise(Logistic; quantiles = (0.25 => 1.0, 0.75 => 3.0)),
        reparameterise(Uniform; quantiles = (0.25 => 1.0, 0.75 => 3.0)),
        reparameterise(Laplace; quantiles = (0.25 => 1.0, 0.75 => 3.0)),
        reparameterise(Exponential; quantiles = (0.95 => 3.0,)))
        nd = native(d)

        @test logpdf(d, 2.0) ≈ logpdf(nd, 2.0)
        @test cdf(d, 2.0) ≈ cdf(nd, 2.0)
        @test quantile(d, 0.4) ≈ quantile(nd, 0.4)
        @test mean(d) ≈ mean(nd)
        @test var(d) ≈ var(nd)
        @test rand(Xoshiro(1), d) == rand(Xoshiro(1), nd)
    end
end
