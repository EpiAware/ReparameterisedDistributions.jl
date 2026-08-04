# As for Enzyme: the solve carries no derivative, the correction supplies
# it. Mooncake otherwise refuses the bitcast in Roots' bisection midpoint.
module ReparameterisedDistributionsMooncakeExt

import ReparameterisedDistributions: _solve_moment_equation
using Mooncake: Mooncake, DefaultCtx, @zero_derivative

@zero_derivative DefaultCtx Tuple{typeof(_solve_moment_equation),
    Any, Any, Any}

end
