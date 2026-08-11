# Enzyme traces into the two solves by dataflow, so these rules mark them
# `inactive` while `solve_moment` and `solve_moments` (`src/numeric.jl`)
# supply the real derivative via an implicit-function-theorem correction;
# each rule must stay paired with that correction or the gradient goes
# silently wrong.
#
# Hessians are not supported through this path under Enzyme; use
# ForwardDiff for a Hessian that touches a numeric conversion.
module ReparameterisedDistributionsEnzymeExt

using ReparameterisedDistributions: _solve_moment_equation,
                                    _solve_moment_system
using Enzyme: Enzyme
using Enzyme.EnzymeRules: EnzymeRules

EnzymeRules.inactive(::typeof(_solve_moment_equation), args...) = nothing
EnzymeRules.inactive(::typeof(_solve_moment_system), args...) = nothing

end
