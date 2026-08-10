# `_solve_moment_equation(f, lo, hi)` is the package extension seam
# (`ReparameterisedDistributionsRootsExt`) that runs the scalar root-find
# for a numeric family's `to_native` (currently only `Weibull(mean, sd)`,
# `src/families.jl`). `solve_moment` (`src/numeric.jl`) calls it on
# `vals` already stripped to its primal via `_primal`, then recovers the
# exact derivative afterwards with two steps of an implicit-function-theorem
# correction: `s` is a root of `residual`, so `s - residual(s, vals) /
# deriv(s, vals)` leaves the VALUE unchanged to machine precision while
# making the DERIVATIVE exactly `-residual_vals / deriv`, regardless of what
# derivative `s` itself carries in.
#
# `_primal` only strips a `ForwardDiff.Dual`/`ReverseDiff.TrackedReal`
# wrapper (see the extensions for those backends); Enzyme carries no such
# wrapper type; it tracks activity by dataflow, not by value type, so
# `_primal`'s identity method on a plain `Real` does not stop Enzyme
# statically tracing INTO `_solve_moment_equation` and from there into
# `Roots.find_zero`'s own internals — where it aborts with
# `Enzyme.Compiler.IllegalTypeAnalysisException` (measured directly against
# the Weibull moment equation, on both `Enzyme.Forward` and `Enzyme.Reverse`).
# Marking the call `EnzymeRules.inactive` runs it on the primal unchanged and
# treats the returned root as `Const`, so Enzyme never type-analyses Roots'
# internals while the IFT correction above still supplies the real
# derivative afterwards from `vals`, exactly as it does for ForwardDiff.
# `inactive` covers every activity / batch-width / mode permutation
# uniformly, matching `_window_quantile`'s treatment in
# CensoredDistributions.jl and ConvolvedDistributions.jl, and `primal`'s in
# EpiAwareADTools.jl. `args...` (rather than typing `f`/`lo`/`hi`) matches
# any closure `f` a family's `to_native` builds, since the closure's own
# type differs per registered family.
module ReparameterisedDistributionsEnzymeExt

using ReparameterisedDistributions: _solve_moment_equation
using Enzyme: Enzyme
using Enzyme.EnzymeRules: EnzymeRules

EnzymeRules.inactive(::typeof(_solve_moment_equation), args...) = nothing

end
