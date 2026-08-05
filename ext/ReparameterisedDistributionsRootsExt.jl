# Supplies the scalar root-find `src/numeric.jl`'s `_solve_moment_equation`
# stub asks for. `A42()` (Alefeld-Potra-Shi) is bracketing, so it cannot leave
# the registered interval, needs no initial guess and no derivative, and
# converges in roughly 10 residual evaluations.
#
# The core asks only for the root's VALUE: the implicit-function-theorem
# correction in `src/numeric.jl` supplies the derivative afterwards as plain
# arithmetic, so this method does not need to (and does not) propagate AD
# types itself — whatever `lo`/`hi`'s number type is, `find_zero` runs in it
# and hands the primal root straight back.
module ReparameterisedDistributionsRootsExt

import ReparameterisedDistributions: _solve_moment_equation
using Roots: A42, find_zero

function _solve_moment_equation(f::F, lo::Real, hi::Real) where {F}
    return find_zero(f, (lo, hi), A42())
end

end
