# `_solve_moment_equation`'s Mooncake counterpart to the Enzyme rule in
# `ReparameterisedDistributionsEnzymeExt`: Mooncake also traces into the
# root-find, so `@zero_derivative` holds it out of AD while `solve_moment`
# supplies the real derivative via its implicit-function-theorem
# correction. `Any` in the closure position matches every family's own
# closure type.
#
# Hessians are not supported through this path, and worse than Enzyme's
# failure: `SecondOrder(AutoForwardDiff(), AutoMooncake())` over the
# Weibull numeric path aborts the whole process rather than raising a
# catchable exception, so it is deliberately not covered by an automated
# test. Use `AutoForwardDiff` for a Hessian that touches a numeric
# (Weibull) conversion.
module ReparameterisedDistributionsMooncakeExt

using ReparameterisedDistributions: _solve_moment_equation,
                                    _solve_moment_system
using Mooncake: Mooncake

Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(_solve_moment_equation), Any, Real, Real}

# The vector counterpart, held out for the same reason: `solve_moments`
# reinjects the derivative straight afterwards.
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(_solve_moment_system), Any, Tuple}

end
