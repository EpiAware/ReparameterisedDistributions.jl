# Strips a ForwardDiff `Dual` to its primal for the numeric conversion's
# bracket and solve (see `_primal` in src/numeric.jl for why this matters:
# `Roots.A42` is not reliably robust in `Dual` arithmetic, even though the
# resulting DERIVATIVE is unaffected — that comes from the
# implicit-function-theorem correction afterwards, in the caller's own
# type, regardless of what type the solve itself ran in).
#
# Recurses through `value` so a higher-order (nested) `Dual` tag chain
# still reduces to a scalar rather than stopping one level early — the
# same pattern CensoredDistributions.jl uses for the same reason.
module ReparameterisedDistributionsForwardDiffExt

import ReparameterisedDistributions: _primal
using ForwardDiff: Dual, value

_primal(x::Dual) = _primal(value(x))

end
