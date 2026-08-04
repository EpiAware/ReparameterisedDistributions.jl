# The numeric fallback: a `(family, names)` pair with no registered closed
# form, but whose conversion is a well-posed scalar root-find over a
# monotone moment equation, opts into a solver-backed conversion instead of
# `to_native`'s "no reparameterisation is registered" error.
#
# This file owns everything EXCEPT the root-find itself: the trait that
# switches a pair onto the numeric path, the driver that turns a solved
# root into a native distribution, and the derivative rule that keeps the
# whole thing differentiable regardless of what the solver returns. The
# root-find is supplied by a package extension (`_solve_moment_equation`
# below), so no solver ships in this package.

# --- The trait -------------------------------------------------------------
#
# Resolved on the (family, names) PAIR at compile time and stores nothing, so
# `Reparameterised`'s field layout, and the `params` NTuple question, stay
# untouched.

"Marks a `(family, names)` pair as converting through an exact closed form."
struct Analytic end

"Marks a `(family, names)` pair as converting through a numeric root-find."
struct Numeric end

# Every pair starts `Analytic`; a family opts into the numeric path with a
# more specific method returning `Numeric()`, registered alongside its
# `_moment_residual` and friends (see the `Weibull` registration in
# families.jl).
_conversion_kind(::Type{D}, ::Val{names}) where {D, names} = Analytic()

# --- The generic `to_native` delegator --------------------------------------
#
# `to_native`'s generic 3-arg method (defined in Reparameterised.jl)
# dispatches here on the trait. A family's own `to_native` method — an exact
# closed form — is always MORE SPECIFIC than that 3-arg fallback, so ordinary
# dispatch picks it first whenever one is registered: analytical conversions
# always win, with no registry and no priority table to maintain.

function _to_native_fallback(::Analytic, ::Type{D}, ::Val{names},
        vals) where {D, names}
    throw(ArgumentError(
        "no reparameterisation of $(D) by $(collect(names)) is " *
        "registered; the registered parameterisations are listed " *
        "in the package docs"))
end

function _to_native_fallback(::Numeric, ::Type{D}, ::Val{names},
        vals) where {D, names}
    pvals = map(_primal, vals)
    lo, hi = _moment_bracket(D, Val(names), pvals)
    f = s -> _moment_residual(D, Val(names), s, pvals)
    _check_bracket(D, Val(names), vals, f(lo), f(hi))
    s0 = _solve_moment_equation(f, lo, hi)
    s = _implicit_correction(D, Val(names), s0, vals)
    return _from_solution(D, Val(names), s, vals)
end

# --- Stripping AD wrapper types for the solve itself ------------------------
#
# `s0` only ever needs to be a ROOT'S VALUE (see the derivative rule above):
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
# derivative are still evaluated in the caller's own type during
# `_implicit_correction`, so this costs nothing in correctness or in which
# types survive to the returned distribution.
#
# The identity default keeps a plain `Float64`/`Float32` solve unchanged.
# `ReparameterisedDistributionsForwardDiffExt` adds the `Dual`-stripping
# method — the established remedy elsewhere in this organisation (see
# CensoredDistributions.jl) — recursing so a higher-order tag chain still
# reduces to a scalar.
_primal(x::Real) = x

# --- The derivative rule -----------------------------------------------------
#
# The implicit function theorem written as arithmetic rather than as a
# per-backend AD rule. `s0` is a root of `F`, so subtracting `F / F_s` leaves
# the VALUE unchanged to machine precision while making the DERIVATIVE exactly
# `-F_vals / F_s`. The incoming `ds0/dvals` cancels term for term, so this
# holds whatever derivative `s0` arrived with — a garbage one, a zero one, or
# none at all, which is what makes the seam safe against a solver backend that
# returns a bare `Float64` and silently truncates a `Dual`.
#
# Two steps, not one: measured against the Weibull `(mean, sd)` equation, one
# correction step gives a correct gradient but a Hessian wrong by 2% on the
# diagonal and 21% off-diagonal. There is no loop over a convergence
# predicate, no bracket and no bisection here — a fixed two statements — so
# every future solver backend inherits this correctness rather than
# re-deriving it, which is exactly how a backend ends up silently truncating.
function _implicit_correction(::Type{D}, ::Val{names}, s0,
        vals) where {D, names}
    s = s0
    for _ in 1:2
        s = s -
            _moment_residual(D, Val(names), s, vals) /
            _moment_residual_deriv(D, Val(names), s, vals)
    end
    _check_solved(D, Val(names), s, vals)
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
# the existing `_valid_moments` machinery a numeric family adds a method for,
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

function _check_solved(::Type{D}, ::Val{names}, s, vals) where {D, names}
    r = _moment_residual(D, Val(names), s, vals)
    abs(r) <= _moment_atol(float(typeof(r))) && return nothing
    throw(DomainError(vals,
        "the numeric conversion of $(D) by $(collect(names)) did not " *
        "converge: residual $(r) at solved value $(s), " *
        "tolerance $(_moment_atol(float(typeof(r))))"))
end
