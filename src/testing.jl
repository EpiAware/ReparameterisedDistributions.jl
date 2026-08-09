# The interface test suite: one call that checks a (family, parameter-name)
# pair against the whole registration contract, so this package's own
# families and a family registered from another package are held to the
# same standard by the same code.
#
# The implementation lives in a package extension, so `Test` stays a weak
# dependency: the helper is only ever called from a test suite, which has
# `Test` loaded already. The stub below is the fallback that extension's
# more specific method takes over from by ordinary dispatch, exactly as
# `_solve_moment_equation`'s stub is taken over by the Roots extension.

@doc raw"

Check a registered (family, parameter-name) pair against the whole
registration contract.

Runs one `@testset` covering everything a registration has to get right:
that it is registered at all and under the canonical (alphabetically
sorted) parameter names; that [`valid_moments`](@ref) answers `Bool` and
answers `true` at the point given; that [`to_native`](@ref) is inferred to
a concrete distribution of the family and never admits `nothing`; that the
wrapper builds, reports the given parameters as its own, keeps the
family's variate form and value support, and evaluates and samples as the
native distribution does; that any parameter named `mean`, `sd` or `var`
comes back out of the built distribution; and that each invalid point
given is rejected at construction and yields `-Inf` without it.

Written for a family registered from another package as much as for this
one's own: it imports nothing private, so a downstream registration can be
held to the same contract by the same code.

`Test` supplies this through a package extension, so a test suite reaches
it with `using Test`. Calling it without `Test` loaded raises.

# Arguments
- the native family the registration converts to.
- `names`: the alternative parameter names, as a tuple of `Symbol`s, in the
  order they are registered under.
- `vals`: valid alternative parameter values, in `names` order.
- `invalid`: alternative parameter tuples that must be rejected. Empty by
  default, but passing at least one is what checks that the family's guard
  actually guards.
- `rtol`: the relative tolerance the `mean`/`sd`/`var` round-trip is held
  to. Loose enough by default for a numeric conversion.

# Examples
```@example
using ReparameterisedDistributions, Distributions, Test
using ReparameterisedDistributions: test_reparameterisation

test_reparameterisation(Gamma, (:mean, :sd), (8.0, 3.0);
    invalid = ((8.0, -1.0), (-8.0, 1.0)))
```

# See also
- [`to_native`](@ref): one of the two hooks this checks.
- [`valid_moments`](@ref): the other.
"
function test_reparameterisation(::Type{D}, names, vals; kwargs...) where {D}
    throw(ArgumentError(
        "the reparameterisation interface test suite lives in a package " *
        "extension; `using Test` loads " *
        "ReparameterisedDistributionsTestExt, which supplies it"))
end
