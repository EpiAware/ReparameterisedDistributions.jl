# `_solve_moment_equation`'s Mooncake counterpart to the Enzyme rule in
# `ReparameterisedDistributionsEnzymeExt` — see that file for the full
# derivation of why the implicit-function-theorem correction in
# `solve_moment` (`src/numeric.jl`) still recovers the exact derivative once
# this call is held out of AD. Measured directly against the Weibull moment
# equation, Mooncake traces INTO `_solve_moment_equation` on both
# `AutoMooncake` and `AutoMooncakeForward` and aborts with an `ArgumentError`
# ("not permissible to bitcast to a differentiable type during AD") from
# `Roots.find_zero`'s own internals.
#
# `@zero_derivative` (no mode argument: covers both ForwardMode and
# ReverseMode) registers the primitive and generates a zero `frule!!` and a
# zero `rrule!!`, so the root's VALUE still comes back correct (the primal
# call runs unchanged) while its derivative is cut — exactly what
# `EnzymeRules.inactive` does for Enzyme, and what the `_primal` strip
# already does for ForwardDiff/ReverseDiff (`src/numeric.jl`). `Any` in the
# closure position (rather than a concrete type) matches every family's own
# `f` closure, since each registered family builds a differently-typed one;
# mirrors the `Any` ctor position in CensoredDistributions'
# `_ctor_has_check_args` Mooncake rule for the same reason.
module ReparameterisedDistributionsMooncakeExt

using ReparameterisedDistributions: _solve_moment_equation
using Mooncake: Mooncake

Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(_solve_moment_equation), Any, Real, Real}

end
