# The numeric seam: a family with no exact closed form still registers
# exactly one `to_native` method and one `valid_moments` method, the same
# contract as an analytic family, and calls `solve_moment` inside its own
# `to_native` body to run a scalar root-find over a monotone moment
# equation. No trait, no registry: an unregistered pair reaches the 3-arg
# `to_native` fallback in Reparameterised.jl by ordinary dispatch.
#
# This file owns the driver (`solve_moment`), the derivative rule that
# keeps the whole thing differentiable regardless of what the solver
# returns, and the three ways a numeric conversion fails cleanly. The
# root-find itself is supplied by a package extension
# (`_solve_moment_equation` below), so no solver ships in this package.

# --- Stripping AD wrapper types for the solve itself ------------------------
#
# `s0` only ever needs to be a ROOT'S VALUE (see the derivative rule below):
# the correction recovers the exact derivative afterwards from whatever
# `vals` the caller actually passed, whatever type `s0` arrived as. That
# safety property is exploited here, not merely tolerated: the bracket and
# the solve itself run on `vals` stripped to its PRIMAL — plain `Float64` (or
# `Float32`, …) — rather than in the caller's own AD type.
#
# This is a robustness choice, not a correctness one, and it is not
# optional in practice. Measured directly against this package's own
# Weibull equation: `Roots.A42` on a `ForwardDiff.Dual`-valued bracket
# converges for many (mean, sd) pairs, but not all — `mean = 74.916,
# sd = 1.079` (arising from an ordinary NUTS trajectory, not a contrived
# edge case) throws `Roots.ConvergenceFailed` in `Dual` arithmetic while
# solving instantly in `Float64`. `!=` on a `Dual` compares partials as
# well as the value, so an internal stall/no-progress check written
# against `Dual` equality can fail to fire even once the VALUE has
# converged, and the iteration runs out its budget. The residual and its
# derivative are still evaluated in the caller's own type during the
# correction below, so this costs nothing in correctness or in which
# types survive to the returned distribution.
#
# The identity default keeps a plain `Float64`/`Float32` solve unchanged.
# `ReparameterisedDistributionsForwardDiffExt` adds the `Dual`-stripping
# method — the established remedy elsewhere in this organisation (see
# CensoredDistributions.jl) — recursing so a higher-order tag chain still
# reduces to a scalar.
_primal(x::Real) = x

@doc raw"

Solve a family's own moment equation for a scalar `s`, exactly and
differentiably, regardless of which solver backend runs the root-find.

A numeric family calls this inside its own [`to_native`](@ref) method,
passing its own `residual`, `deriv` and `bracket` as ordinary functions —
this registers no method on anything of ours, so a downstream package
supplies its own three functions exactly as this package's own `Weibull`
registration does (see `src/families.jl`).

- `residual(s, vals)` is the moment equation, zero at the solution.
- `deriv(s, vals)` is its derivative with respect to `s`.
- `bracket(pvals) -> (lo, hi)` is a sign-changing interval for `residual`,
  evaluated on `vals` stripped to its primal type.

The root-find itself runs on the primal-stripped `vals`, then two steps of
an implicit-function-theorem correction recover the exact derivative in
the caller's own type afterwards: `s` is a root of `residual`, so
subtracting `residual / deriv` leaves the VALUE unchanged to machine
precision while making the DERIVATIVE exactly `-residual_vals / deriv`,
whatever derivative the solver's own `s` arrived with (a garbage one, a
zero one, or none at all). Two steps, not one: measured against the
Weibull `(mean, sd)` equation, one correction step gives a correct
gradient but a Hessian wrong by 2% on the diagonal and 21%
off-diagonal.

# Arguments
- the native family being converted to.
- `Val(names)`: the alternative parameter names.
- `residual`, `deriv`, `bracket`: the family's own moment equation.
- `vals`: the alternative parameter values, in `names` order.

# Examples
The `to_native` method that calls this must be registered under the
parameter names sorted alphabetically, as `reparameterise` canonicalises
its keywords that way before dispatching.

`solve_moment` is public but not exported, so import it by name.

```@example
using ReparameterisedDistributions, Distributions
using ReparameterisedDistributions: solve_moment

# A toy equation, exp(s) = 2, solved for illustration; a real family's
# residual is its own moment equation.
solve_moment(Gamma, Val((:shape,)), (s, vals) -> exp(s) - vals[1],
    (s, vals) -> exp(s), pvals -> (-10.0, 10.0), (2.0,))
```

# See also
- [`to_native`](@ref): the conversion a numeric family calls this from.
- [`valid_moments`](@ref): the guard a numeric family also needs.
"
function solve_moment(::Type{D}, ::Val{names}, residual::R, deriv::G,
        bracket::B, vals) where {D, names, R, G, B}
    pvals = map(_primal, vals)
    lo, hi = bracket(pvals)
    f = s -> residual(s, pvals)
    _check_bracket(D, Val(names), vals, f(lo), f(hi))
    s = _solve_moment_equation(f, lo, hi)
    for _ in 1:2
        s = s - residual(s, vals) / deriv(s, vals)
    end
    _check_solved(D, Val(names), residual, s, vals)
    return s
end

# --- The solver seam ---------------------------------------------------------
#
# The only thing this package does not do itself. A package extension adds a
# strictly more specific method (`lo::Real, hi::Real` beats the `Any, Any`
# here), so loading it takes over by ordinary dispatch — no registry, no
# mutable global, no `hasmethod` at run time, no method-overwrite warning.
# Deliberately no solver-type argument and no `solver =` keyword: with a
# single backend shipped that would be a knob nobody threads through.
function _solve_moment_equation(f, lo, hi)
    throw(ArgumentError(
        "this parameterisation is converted by a scalar root-find, " *
        "and no solver backend is loaded; add Roots.jl to your " *
        "project and `using Roots` to load " *
        "ReparameterisedDistributionsRootsExt, which supplies it"))
end

# --- The three ways a numeric conversion fails cleanly -----------------------
#
# (a), moments outside a family's numerically solvable window, is handled by
# the existing `valid_moments` machinery a numeric family adds a method for,
# unchanged in kind from the analytical families. (b) and (c) are below.

# (b) The moment equation does not change sign over the registered bracket:
# throw before the solver runs, so the message names the family and the
# working type rather than surfacing Roots' generic bracketing complaint.
function _check_bracket(::Type{D}, ::Val{names}, vals, flo,
        fhi) where {D, names}
    (isfinite(flo) && isfinite(fhi) && flo * fhi < 0) && return nothing
    throw(DomainError(vals,
        "the moment equation for $(D) by $(collect(names)) does not " *
        "change sign over the registered bracket, so these moments " *
        "are outside the numerically solvable region for " *
        "$(eltype(vals))"))
end

# (c) The solve ran, but the corrected root misses tolerance: throw rather
# than return a distribution whose moments differ from what was asked for.
_moment_atol(::Type{T}) where {T} = sqrt(eps(T))

function _check_solved(::Type{D}, ::Val{names}, residual::R, s,
        vals) where {D, names, R}
    r = residual(s, vals)
    abs(r) <= _moment_atol(float(typeof(r))) && return nothing
    throw(DomainError(vals,
        "the numeric conversion of $(D) by $(collect(names)) did not " *
        "converge: residual $(r) at solved value $(s), " *
        "tolerance $(_moment_atol(float(typeof(r))))"))
end
