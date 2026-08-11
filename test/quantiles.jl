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
    @test valid_moments(Normal, Val((:rate, :shape)), (1.0, 2.0)) == true
    @test_throws ArgumentError reparameterise(Normal; rate = 1.0, shape = 2.0)
end

@testitem "standard moments: the location-scale families, exact" begin
    using Distributions

    # Against native parameters worked by hand from each family's own
    # variance, not against the constraint solve that produces them.
    m, s = 5.0, 2.0
    cases = ((Normal, Normal(m, s)),
        # var = theta^2 * pi^2 / 3, so theta = sd * sqrt(3) / pi.
        (Logistic, Logistic(m, s * sqrt(3) / pi)),
        # var = 2 * theta^2, so theta = sd / sqrt(2).
        (Laplace, Laplace(m, s / sqrt(2))),
        # var = (b - a)^2 / 12 about a midpoint mean, so the half-width is
        # sqrt(3) * sd.
        (Uniform, Uniform(m - sqrt(3) * s, m + sqrt(3) * s)))

    for (D, expected) in cases
        d = reparameterise(D; mean = m, sd = s)
        @test native(d) ≈ expected
        @test mean(d)≈m rtol=1e-12
        @test std(d)≈s rtol=1e-12
        @test params(d) == (m, s)

        # The variance spelling is the same conversion at sqrt(var).
        v = reparameterise(D; mean = m, var = s^2)
        @test native(v) ≈ expected
        @test params(v) == (m, s^2)
    end
end

@testitem "standard moments: Exponential by its mean" begin
    using Distributions

    # An Exponential(theta) has mean = theta, so the mean is the scale.
    d = reparameterise(Exponential; mean = 4.0)
    @test native(d) ≈ Exponential(4.0)
    @test mean(d)≈4.0 rtol=1e-12
    @test params(d) == (4.0,)

    # The same distribution the rate spelling gives.
    @test native(d) ≈ native(reparameterise(Exponential; rate = 0.25))

    @test_throws DomainError reparameterise(Exponential; mean = -1.0)
    @test_throws DomainError reparameterise(Exponential; mean = 0.0)
end

@testitem "standard moments: the moments that pin nothing are refused" begin
    using Distributions

    # A Cauchy has neither moment, so no Cauchy has the ones asked for.
    @test_throws ArgumentError reparameterise(Cauchy; mean = 1.0, sd = 2.0)
    @test_throws ArgumentError reparameterise(Cauchy; mean = 1.0, var = 4.0)
    try
        reparameterise(Cauchy; mean = 1.0, sd = 2.0)
    catch e
        @test occursin("Cauchy", sprint(showerror, e))
    end

    # One constraint leaves a two-parameter family short.
    for D in (Normal, Logistic, Laplace, Uniform, Cauchy, LogNormal)
        @test_throws ArgumentError reparameterise(D; mean = 1.0)
    end

    # And an Exponential has sd == mean, so both together are one too many.
    @test_throws ArgumentError reparameterise(Exponential; mean = 1.0,
        sd = 2.0)
    @test_throws ArgumentError reparameterise(Exponential; mean = 1.0,
        var = 4.0)
end

@testitem "quantiles: the derived scale has to survive, not just the values" begin
    using Distributions
    using ReparameterisedDistributions: valid_moments

    # #93/#94 in constraint coordinates: each value is finite and ordered,
    # yet the scale the pair implies is not a scale any member has.
    #
    # Far enough apart and the scale overflows.
    @test valid_moments(Normal, Val((0.25, 0.75)),
        (-1e308, 1e308)) == false
    @test logpdf(
        reparameterise(
            Normal; quantiles = (0.25 => -1e308,
                0.75 => 1e308), check_args = false),
        0.0) == -Inf

    # Close enough together and it collapses to zero.
    @test valid_moments(Normal, Val((0.25, 0.75)), (1.0, 1.0)) == false

    # A Uniform carries the width in its second native parameter, so the
    # quantity that has to survive is the far edge. A width too small to
    # register against its own location gives `a == b`, which is no
    # Uniform even though the scale itself is positive — and the same
    # numbers are a perfectly ordinary Normal, so the check has to be the
    # family's own rather than shared.
    @test valid_moments(Uniform, Val((:mean, :sd)), (1e300, 1e270)) == false
    @test valid_moments(Normal, Val((:mean, :sd)), (1e300, 1e270)) == true
end

@testitem "quantiles: the registration table says what it claims" begin
    using Distributions
    import ReparameterisedDistributions as RD

    # The table every constraint set is checked against. It is read from
    # inside a `@generated` body, so nothing reaches it at run time and a
    # wrong entry would surface only as a puzzling refusal somewhere else.
    #
    # Held in `Any` containers so each lookup is a real dynamic call. A
    # family named as a literal is a compile-time constant, and these
    # methods return constants, so the compiler folds the answer and the
    # method never actually runs — which is also why the table reads as
    # uncovered when it is asserted the direct way.
    two = Any[Normal, LogNormal, Logistic, Cauchy, Uniform, Laplace]
    for D in two
        @test RD._constraint_arity(D) == 2
        # The arity is the family's native parameter count, not a number
        # chosen to suit the solve.
        @test length(fieldnames(D)) == 2
    end
    @test RD._constraint_arity(Any[Exponential][1]) == 1
    @test length(fieldnames(Exponential)) == 1

    # A family with no exact inversion has no arity, which is what sends a
    # quantile request to the seam rather than to a wrong solve.
    for D in Any[Gamma, Beta, Weibull, InverseGaussian, NegativeBinomial]
        @test RD._constraint_arity(D) === nothing
        @test RD._moment_names(D) == ()
    end

    # The moment names each inversion can express. A Cauchy has neither
    # moment and a LogNormal's are linear in neither native parameter, so
    # both offer the median alone.
    for D in Any[Cauchy, LogNormal]
        @test RD._moment_names(D) == (:median,)
    end
    # An Exponential's standard deviation is its scale, so it is a row here
    # too; its mean is the whole constraint set and is registered
    # concretely instead.
    @test RD._moment_names(Any[Exponential][1]) == (:median, :sd)
    for D in Any[Normal, Logistic, Uniform, Laplace]
        @test RD._moment_names(D) == (:mean, :median, :sd)
    end

    # The standard member's own standard deviation, which is what turns an
    # `sd` constraint into a row in the scale. Checked against each
    # family's own `std` rather than restated as a formula.
    for (D, standard) in Any[(Normal, Normal(0.0, 1.0)),
        (Logistic, Logistic(0.0, 1.0)), (Laplace, Laplace(0.0, 1.0)),
        (Uniform, Uniform(0.0, 1.0))]
        @test RD._std_sd(D) ≈ std(standard)
    end

    # Only a LogNormal restricts where its constraint values may fall, and
    # it does so because the log transform is taken before the solve.
    for (D, v, expected) in Any[(LogNormal, 1.0, true),
        (LogNormal, -1.0, false), (LogNormal, 0.0, false),
        (Normal, -1.0, true), (Uniform, -1.0, true)]
        @test RD._in_domain(D, v) === expected
    end
end

@testitem "quantiles: a refused pair raises rather than reading zero" begin
    using Distributions
    using ReparameterisedDistributions: valid_moments

    # A pair registered only to refuse keeps its predicate `true`, so the
    # conversion's own error is what a caller sees. `false` would instead
    # make a meaningless request read as a merely improbable one, giving a
    # silent `-Inf` where an error belongs.
    # `Any` so each predicate is a real dynamic call rather than a constant
    # the compiler folds away before the method runs.
    refused = Any[(Cauchy, (:mean, :sd), (1.0, 2.0)),
        (Cauchy, (:mean, :var), (1.0, 4.0)),
        (Exponential, (:mean, :sd), (1.0, 2.0)),
        (Exponential, (:mean, :var), (1.0, 4.0)),
        (Normal, (:mean,), (1.0,)), (Uniform, (:mean,), (1.0,)),
        (LogNormal, (:mean,), (1.0,))]

    for (D, names, vals) in refused
        @test valid_moments(D, Val(names), vals) === true
        # And `check_args = false` does not turn the refusal into a zero
        # density: the request is meaningless, not improbable.
        d = ReparameterisedDistributions._build(D, Val(names), vals;
            check_args = false)
        @test_throws ArgumentError logpdf(d, 1.0)
    end
end

@testitem "quantiles: Exponential by its median" begin
    using Distributions

    # Q(1/2) = theta * log(2), so the median names the scale too.
    d = reparameterise(Exponential; median = 2.0)
    @test native(d) ≈ Exponential(2.0 / log(2.0))
    @test median(d)≈2.0 rtol=1e-12
    @test params(d) == (2.0,)
end

@testitem "quantiles: a narrower value type is not widened" begin
    using Distributions

    # The probabilities are `Float64` whatever the values are, so the
    # conversion has to bring them into the values' own type rather than
    # promote the wrapper up to `Float64`.
    d = reparameterise(Normal; quantiles = (0.25 => 1.0f0, 0.75 => 3.0f0))
    @test eltype(params(d)) === Float32
    @test native(d) isa Normal{Float32}
end

@testitem "quantiles: rescale routes through a mixed constraint set" begin
    using Distributions

    # `rescale` scales a registered name and rebuilds through the same
    # conversion, so a moment mixed with a quantile is scaled in the
    # constraint coordinates rather than by an affine transform.
    d = reparameterise(Normal; mean = 8.0, quantiles = (0.95 => 20.0,))
    @test mean(rescale(d, 2.0)) ≈ 16.0
    @test quantile(rescale(d, 2.0), 0.95)≈20.0 rtol=1e-12

    # A probability is not a name `rescale` can be asked for.
    @test_throws DomainError rescale(d, 2.0; parameter = :median)
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

@testitem "standard moments: Exponential by its standard deviation" begin
    using Distributions

    # An Exponential's standard deviation is its scale, so one constraint
    # is exactly the right number and the request is determined.
    d = reparameterise(Exponential; sd = 2.0)
    @test native(d) ≈ Exponential(2.0)
    @test std(d)≈2.0 rtol=1e-12
    @test mean(d)≈2.0 rtol=1e-12
    @test params(d) == (2.0,)

    @test_throws DomainError reparameterise(Exponential; sd = -1.0)
end

@testitem "quantiles: constraints that repeat each other are refused" begin
    using Distributions

    # The coefficients of a row depend only on its NAME, so two constraints
    # saying the same thing is a fact about the request rather than about
    # its values: a Normal is symmetric about its location, so its mean and
    # its median are one row written twice and no numbers can rescue the
    # pair. It is the same defect as a repeated probability, so it gets the
    # same structural error rather than a `DomainError` about the values.
    @test_throws ArgumentError reparameterise(Normal; mean = 5.0,
        median = 5.0)
    @test_throws ArgumentError reparameterise(Normal; median = 5.0,
        quantiles = (0.5 => 5.0,))
    @test_throws ArgumentError reparameterise(Uniform; mean = 5.0,
        median = 5.0)

    for f in (() -> reparameterise(Normal; mean = 5.0, median = 5.0),
        () -> reparameterise(Normal; median = 5.0,
        quantiles = (0.5 => 5.0,)))
        try
            f()
        catch e
            msg = sprint(showerror, e)
            @test occursin("not independent", msg)
            @test occursin("Normal", msg)
        end
    end

    # A quantile at one half is the median, so it collides the same way,
    # while two distinct probabilities do not.
    @test reparameterise(Normal; quantiles = (0.25 => 1.0, 0.75 => 3.0)) isa
          Distribution

    # The mean is NOT the median of a family that is not symmetric about
    # its location, so the two are independent there and the pair is a
    # different refusal — one this package leaves to the numeric solve.
    @test_throws ArgumentError reparameterise(LogNormal; mean = 5.0,
        quantiles = (0.95 => 9.0,))
end

@testitem "quantiles: a later registration is not answered from a stale cache" begin
    using Distributions
    import ReparameterisedDistributions as RD

    # `_convertible` reads `_constraint_arity`, `_moment_names` and the
    # constraint rows. Were it a generated function its answer would be
    # cached per signature and never invalidated, so a family registered
    # AFTER the first call would keep the stale `false` for the rest of the
    # session, with no error to say so.
    #
    # A local type rather than a real family: this fixes a dispatch
    # property, and says nothing about anyone's distribution.
    struct StaleProbe <: ContinuousUnivariateDistribution end

    # Nothing is registered for it, so it answers `false` — and the answer
    # is cached at this point under a generated implementation.
    @test RD._convertible(StaleProbe, Val((:median, 0.75))) == false

    # Register it, exactly as the location-scale families are registered.
    RD._constraint_arity(::Type{StaleProbe}) = 2
    RD._moment_names(::Type{StaleProbe}) = (:median,)
    RD._std_quantile(::Type{StaleProbe}, p) = log(p / (1 - p))

    # The answer has to follow the method table, which is what a cached
    # generator would not do.
    @test RD._convertible(StaleProbe, Val((:median, 0.75))) == true

    # And the arity registered is the one enforced, rather than one baked
    # in at the first call.
    @test RD._convertible(StaleProbe, Val((:median,))) == false
end
