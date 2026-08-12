# Supplies the interface test suite `src/testing.jl`'s
# `test_reparameterisation` stub asks for. It lives here rather than in the
# package proper so `Test` stays a weak dependency: the only caller is a
# test suite, which has `Test` loaded already.
#
# Everything below goes through the public surface — `reparameterise`,
# `native`, `to_native`, `valid_moments` and the Distributions.jl interface
# — so a family registered from another package is checked by exactly the
# same code as one registered here.
module ReparameterisedDistributionsTestExt

import ReparameterisedDistributions: test_reparameterisation
using ReparameterisedDistributions: ReparameterisedDistributions, native,
                                    reparameterise, to_native, valid_moments
using Distributions: Distributions, Distribution, logpdf, mean, median, params,
                     pdf, quantile, std, var
using Random: Xoshiro
using Test: @inferred, @test, @test_throws, @testset

# A family nothing registers, so `which` on it always lands on the
# three-argument fallback. Probing with the family under test instead would
# find that family's own catch-all method, where it has one — every family
# reparameterised by quantile constraints does, since a probability cannot
# be dispatched on.
struct _NeverRegistered end

# The keyword call `reparameterise` would be given for this constraint set:
# moment names as their own keywords, probabilities gathered into
# `quantiles`.
function _rebuild(::Type{D}, names, vals; check_args::Bool = true) where {D}
    moments = NamedTuple{Tuple(n for n in names if n isa Symbol)}(Tuple(
        v for (n, v) in zip(names, vals) if n isa Symbol))
    qs = Tuple(n => v for (n, v) in zip(names, vals) if !(n isa Symbol))
    isempty(qs) && return reparameterise(D; check_args = check_args, moments...)
    return reparameterise(D; check_args = check_args, quantiles = qs,
        moments...)
end

function test_reparameterisation(
        ::Type{D}, names::Tuple{Vararg{Union{Symbol, Real}}},
        vals::Tuple{Vararg{Real}}; invalid = (),
        rtol::Real = 1.0e-6) where {D}
    length(names) == length(vals) || throw(ArgumentError(
        "expected one value per parameter name, got $(length(names)) " *
        "names and $(length(vals)) values"))

    # Normalise exactly as `reparameterise` does internally, so every check
    # below runs on the values the registered methods are actually called
    # with rather than on whatever the caller happened to type.
    pvals = promote(map(float, vals)...)
    argtypes = (Type{D}, Val{names}, typeof(pvals))

    @testset "reparameterise($(D); $(join(names, ", ")))" begin
        # Registered, and under the canonical name order. `reparameterise`
        # canonicalises its constraint set before dispatching, so a method
        # registered under any other order is never found.
        @test names ==
              first(ReparameterisedDistributions._canonical(Val(names), pvals))
        # The 3-arg fallback. A registration that dispatches to it is not
        # registered at all.
        fallback = which(to_native,
            (Type{_NeverRegistered}, Val{(:_unregistered_,)}, typeof(pvals)))
        @test which(to_native, argtypes) !== fallback

        # The guard answers `Bool`, and answers `true` here. Every call
        # site branches on this before converting, so a non-`Bool` answer
        # puts the branch's own type in doubt.
        v = valid_moments(D, Val(names), pvals)
        @test v isa Bool
        @test v === true

        # The conversion is inferred to a concrete distribution of the
        # family, and never admits `nothing`. This is the invariant #86
        # rests on: a call site that bound a `Union{Nothing, D}`-typed
        # conversion result gave a silently wrong reverse-mode gradient.
        # `@inferred` on the density cannot see it — union-splitting
        # narrows the outer return type back to a concrete `Float64` on
        # every branch — so it is checked here, on the conversion itself.
        rts = Base.return_types(to_native, argtypes)
        @test length(rts) == 1
        rt = length(rts) == 1 ? only(rts) : Any
        @test rt <: D
        @test isconcretetype(rt)
        @test !(Nothing <: rt)

        # The wrapper builds, and the alternative parameters are its
        # parameters.
        d = _rebuild(D, names, pvals)
        @test d isa Distribution
        @test params(d) == pvals

        # The variate form and value support come from the family, so a
        # discrete family does not silently become continuous.
        @test Distributions.value_support(typeof(d)) ==
              Distributions.value_support(D)
        @test Distributions.variate_form(typeof(d)) ==
              Distributions.variate_form(D)

        # The conversion runs, lands where inference said it would, and is
        # itself inferred through the wrapper.
        nd = @inferred native(d)
        @test nd isa D
        @test typeof(nd) === rt

        # It evaluates and samples as the native distribution does, and the
        # densities stay inferred: these are the sampler's hot path.
        x = rand(Xoshiro(1), nd)
        @test logpdf(d, x) ≈ logpdf(nd, x)
        @test pdf(d, x) ≈ pdf(nd, x)
        @test (@inferred logpdf(d, x)) isa Real
        @test (@inferred pdf(d, x)) isa Real
        @test rand(Xoshiro(1), d) == rand(Xoshiro(1), nd)

        # Any parameter named for a moment, and every elicited quantile,
        # must come back out of the built distribution, which is what
        # checks the conversion's algebra rather than merely its types. A
        # name this package has no reading of (`shape`, `rate`,
        # `mass_below_centre`, …) is left alone.
        for (n, val) in zip(names, pvals)
            if n === :mean
                @test mean(d)≈val rtol=rtol
            elseif n === :sd
                @test std(d)≈val rtol=rtol
            elseif n === :var
                @test var(d)≈val rtol=rtol
            elseif n === :median
                @test median(d)≈val rtol=rtol
            elseif n isa Real
                @test quantile(d, n)≈val rtol=rtol
            end
        end

        # Each invalid point is rejected in the alternative parameters' own
        # coordinates, raises at construction, and gives a zero density
        # without the construction check rather than raising mid-gradient.
        for bad in invalid
            pbad = promote(map(float, bad)...)
            @test !valid_moments(D, Val(names), pbad)
            @test_throws DomainError _rebuild(D, names, pbad)
            b = _rebuild(D, names, pbad; check_args = false)
            @test logpdf(b, x) == -Inf
            @test pdf(b, x) == 0
        end
    end
    return nothing
end

end
