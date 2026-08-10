# Enzyme traces into `_solve_moment_equation`'s root-find by dataflow, so
# this rule marks it `inactive` while `solve_moment` (`src/numeric.jl`)
# supplies the real derivative via an implicit-function-theorem
# correction; this rule must stay paired with that correction or the
# gradient goes silently wrong.
#
# Hessians are not supported through this path under Enzyme; use
# ForwardDiff for a Hessian that touches a numeric (Weibull) conversion.
module ReparameterisedDistributionsEnzymeExt

using ReparameterisedDistributions: _solve_moment_equation
using Enzyme: Enzyme
using Enzyme.EnzymeRules: EnzymeRules

EnzymeRules.inactive(::typeof(_solve_moment_equation), args...) = nothing

end
