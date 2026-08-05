# The solve is a constant as far as Enzyme is concerned: the
# implicit-function-theorem correction in src/numeric.jl re-derives the
# derivative afterwards in the caller's own type. Without this, Enzyme
# traces into Roots' bisection midpoint (a float/integer bitcast) and
# fails type analysis in both modes.
module ReparameterisedDistributionsEnzymeCoreExt

import ReparameterisedDistributions: _solve_moment_equation
using EnzymeCore: EnzymeRules

EnzymeRules.inactive(::typeof(_solve_moment_equation), args...) = nothing

end
