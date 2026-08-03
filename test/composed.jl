# Leaf-coordinate contract with ComposedDistributions: a `Reparameterised`
# leaf reports its registered moments (not the native parameters) as its
# estimable parameters, and rebuilds in those coordinates. Tagged `[:composed]`
# so this file's items are identifiable as depending on the optional
# extension, mirroring `[:turing]` in test/turing.jl.

@testitem "Composed: extension actually loads" tags=[:composed] begin
    using ComposedDistributions

    ext = Base.get_extension(
        ReparameterisedDistributions,
        :ReparameterisedDistributionsComposedDistributionsExt)
    @test ext !== nothing
end

@testitem "Composed: param_names reports the registered moments" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    @test ComposedDistributions.param_names(d) == (:mean, :sd)
    @test ComposedDistributions.leaf_param_names(d) == (:mean, :sd)

    # A three-name registration.
    sn = reparameterise(SkewNormal;
        centre = 0.0, mass_below_centre = 0.4, scale = 1.0)
    @test ComposedDistributions.param_names(sn) ==
          (:centre, :mass_below_centre, :scale)

    # A discrete family registration.
    nb = reparameterise(NegativeBinomial; mean = 10.0, overdispersion = 0.5)
    @test ComposedDistributions.param_names(nb) == (:mean, :overdispersion)

    # A rate-parameterised family (a two-name registration on Gamma).
    g = reparameterise(Gamma; rate = 0.5, shape = 2.0)
    @test ComposedDistributions.param_names(g) == (:rate, :shape)
end

@testitem "Composed: params_table reports moments, not native params" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    tree = compose((delay = d, tail = LogNormal(0.5, 0.4)))
    tbl = params_table(tree)

    # Assert on the columns, not the row count, so a regression to
    # `:param_1`/`:param_2` fails loudly rather than merely miscounting.
    @test collect(tbl.param) == [:mean, :sd, :mu, :sigma]
    @test collect(tbl.value) == [8.0, 2.0, 0.5, 0.4]
end

@testitem "Composed: uncertain places a prior on the mean" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    prior = LogNormal(log(8.0), 0.2)
    u = uncertain(d; mean = prior)
    tree = compose((delay = u, tail = LogNormal(0.5, 0.4)))
    tbl = params_table(tree)

    idx = findall(==(:delay), collect(tbl.edge))
    @test collect(tbl.param)[idx] == [:mean, :sd]
    @test tbl.prior[idx[1]] === prior
    @test tbl.prior[idx[2]] === nothing
end

@testitem "Composed: build_priors puts the prior on the moment" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    prior = LogNormal(log(8.0), 0.2)
    u = uncertain(d; mean = prior)
    tree = compose((delay = u, tail = LogNormal(0.5, 0.4)))

    priors = build_priors(params_table(tree))
    @test priors.delay.mean === prior
    # The `sd` row has no attached prior, so it gets its default (derived
    # from support), not the native-parameter default.
    @test priors.delay.sd isa Distribution
end

@testitem "Composed: uncertain re-pinning rebuilds through leaf_ctor" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    u = uncertain(d; mean = 9.0, sd = LogNormal(0.0, 1.0))
    @test u isa ComposedDistributions.Uncertain
    @test params(u.template) == (9.0, 2.0)
end

@testitem "Composed: update rebuilds a Reparameterised leaf" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    tree = compose((delay = d, tail = LogNormal(0.5, 0.4)))
    updated = ComposedDistributions.update(tree,
        (delay = (mean = 10.0, sd = 3.0), tail = (mu = 0.5, sigma = 0.4)))

    leaf = updated.components[1]
    # The type assertion is the point: it is what catches a stray `free_leaf`
    # method silently collapsing the leaf to a bare `LogNormal`.
    @test leaf isa ReparameterisedDistributions.Reparameterised{
        LogNormal, (:mean, :sd)}
    @test params(leaf) == (10.0, 3.0)
    @test native(leaf) ==
          native(reparameterise(LogNormal; mean = 10.0, sd = 3.0))
end

@testitem "Composed: a discrete family survives the round trip" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    nb = reparameterise(NegativeBinomial; mean = 10.0, overdispersion = 0.5)
    tree = compose((a = nb,))
    updated = ComposedDistributions.update(tree,
        (a = (mean = 12.0, overdispersion = 0.4),))
    leaf = updated.components[1]

    @test leaf isa DiscreteUnivariateDistribution
    @test mean(leaf) ≈ 12.0
end

@testitem "Composed: mixed-type update stays concretely typed" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    tree = compose((delay = d, tail = LogNormal(0.5, 0.4)))
    updated = ComposedDistributions.update(tree,
        (delay = (mean = 10, sd = 3), tail = (mu = 0.5, sigma = 0.4)))

    leaf = updated.components[1]
    @test typeof(params(leaf)) == NTuple{2, Float64}
end

@testitem "Composed: leaf_ctor is egal-stable across identical leaves" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    a = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    b = reparameterise(LogNormal; mean = 1.0, sd = 1.0)
    ctor_a = ComposedDistributions.leaf_ctor(a)
    ctor_b = ComposedDistributions.leaf_ctor(b)
    @test ctor_a === ctor_b

    # A different family.
    g = reparameterise(Gamma; mean = 8.0, shape = 2.0)
    @test ctor_a !== ComposedDistributions.leaf_ctor(g)

    # A different registered parameterisation of the same family.
    g_rate = reparameterise(Gamma; rate = 0.5, shape = 2.0)
    @test ComposedDistributions.leaf_ctor(g) !==
          ComposedDistributions.leaf_ctor(g_rate)
end

@testitem "Composed: tied identical leaves are inventoried once" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    a = shared(:inc, reparameterise(LogNormal; mean = 8.0, sd = 2.0))
    b = shared(:inc, reparameterise(LogNormal; mean = 8.0, sd = 2.0))
    tree = compose((x = a, y = b))

    tbl = params_table(tree)
    @test collect(tbl.edge) == [:inc, :inc]

    updated = ComposedDistributions.update(tree,
        (inc = (mean = 11.0, sd = 4.0),))
    @test params(updated.components[1]) == (11.0, 4.0)
    @test params(updated.components[2]) == (11.0, 4.0)
end

@testitem "Composed: keyword order is canonicalised across the rebuild" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    ordered = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    reordered = reparameterise(LogNormal; sd = 2.0, mean = 8.0)

    tree1 = compose((a = ordered,))
    tree2 = compose((a = reordered,))
    @test collect(params_table(tree1).param) ==
          collect(params_table(tree2).param)
    @test ComposedDistributions.leaf_ctor(ordered) ===
          ComposedDistributions.leaf_ctor(reordered)
end

@testitem "Composed: uncertain on a native parameter throws" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    @test_throws ArgumentError uncertain(d; sigma = LogNormal(0.0, 1.0))

    msg = sprint(showerror,
        try
            uncertain(d; sigma = LogNormal(0.0, 1.0))
        catch e
            e
        end)
    @test occursin("sigma", msg)
    @test occursin("[:mean, :sd]", msg)
end

@testitem "Composed: update on a native parameter throws ArgumentError" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    tree = compose((delay = d, tail = LogNormal(0.5, 0.4)))
    @test_throws ArgumentError ComposedDistributions.update(tree,
        (delay = (mu = 2.0, sigma = 0.2), tail = (mu = 0.5, sigma = 0.4)))
end

@testitem "Composed: uncertain rejects a non-Real non-distribution value" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    @test_throws ArgumentError uncertain(d; mean = "8")
end

@testitem "Composed: a Truncated leaf reports/rebuilds in moments" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    d = truncated(reparameterise(LogNormal; mean = 8.0, sd = 2.0);
        upper = 20.0)
    tree = compose((delay = d,))
    tbl = params_table(tree)

    # `param_names`/`leaf_ctor` are read through `free_leaf`, so the
    # Truncated wrapper must not shadow the registered moment names with
    # the native `mu`/`sigma`, nor with positional `:param_1`/`:param_2`.
    @test collect(tbl.param) == [:mean, :sd]
    @test collect(tbl.value) == [8.0, 2.0]

    updated = ComposedDistributions.update(tree,
        (delay = (mean = 10.0, sd = 3.0),))
    leaf = updated.components[1]

    # `rewrap_leaf` must restore the Truncated wrapper around the rebuilt
    # Reparameterised leaf, not return the bare rebuilt leaf.
    @test leaf isa Truncated
    @test leaf.upper == 20.0
    @test leaf.untruncated isa ReparameterisedDistributions.Reparameterised{
        LogNormal, (:mean, :sd)}
    @test params(leaf.untruncated) == (10.0, 3.0)
end

@testitem "Composed: update propagates DomainError for invalid moments" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    tree = compose((delay = d, tail = LogNormal(0.5, 0.4)))
    @test_throws DomainError ComposedDistributions.update(tree,
        (delay = (mean = 8.0, sd = -2.0), tail = (mu = 0.5, sigma = 0.4)))

    nb = reparameterise(NegativeBinomial; mean = 10.0, overdispersion = 0.5)
    nb_tree = compose((a = nb,))
    @test_throws DomainError ComposedDistributions.update(nb_tree,
        (a = (mean = 10.0, overdispersion = 0.0),))
end

@testitem "Composed: flat codec round-trip (upstream-blocked)" tags=[
    :composed] begin
    using Distributions, ComposedDistributions

    # `_params_arity_of`/`_param_names_of`, the flat codec's type-level
    # companions, are not extensible from outside ComposedDistributions (a
    # `@generated`-function world-age constraint), so the codec still reads
    # `fieldcount(Reparameterised) == 1` and `()` rather than this leaf's
    # registered moments. See the upstream issue opened alongside this
    # extension. These are written `@test_broken`, not omitted, so the suite
    # flips to a visible failure the moment upstream lands a fix.
    d = reparameterise(LogNormal; mean = 8.0, sd = 2.0)
    prior = LogNormal(log(8.0), 0.2)
    u = uncertain(d; mean = prior)
    tree = compose((delay = u, tail = LogNormal(0.5, 0.4)))

    @test_broken ComposedDistributions.flat_dimension(tree) == 1
    @test_broken ComposedDistributions.unflatten(tree, [9.0]) ==
                 (delay = (mean = 9.0, sd = 2.0), tail = NamedTuple())
    @test_broken params(
        ComposedDistributions.reconstruct(tree, [9.0]).components[1]) ==
                 (9.0, 2.0)
    @test_broken ComposedDistributions.flatten(
        tree, ComposedDistributions.unflatten(tree, [9.0])) == [9.0]

    # A loud-failure guard of our own while the above is broken: the known
    # disagreement between the estimable names and the codec's type-level
    # names. The day this assertion itself fails is the day to revisit
    # this file.
    @test ComposedDistributions.leaf_param_names(d) !=
          ComposedDistributions._leaf_type_param_names(typeof(d))
end
