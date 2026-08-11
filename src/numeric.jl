# The numeric seam: a family with no exact closed form still registers the
# same two methods as an analytic family, `valid_moments` and `to_native`,
# and calls `solve_moment` inside its `to_native` body to run a scalar
# root-find over a monotone moment equation. `valid_moments` states its own
# validity — including the window the root-find can actually solve — so the
# solver never runs on an invalid point; `to_native` itself does not guard.
# No trait, no registry: an unregistered pair reaches the 3-arg `to_native`
# fallback in Reparameterised.jl by ordinary dispatch.
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
# optional in practice: a solver's own internal convergence checks can
# fail to fire on a `Dual`-valued bracket even once the value has
# converged, running out the iteration budget on inputs that solve
# instantly in `Float64`. The residual and its derivative are still
# evaluated in the caller's own type during the correction below, so this
# costs nothing in correctness or in which types survive to the returned
# distribution.
#
# The identity default keeps a plain `Float64`/`Float32` solve unchanged.
# `ReparameterisedDistributionsForwardDiffExt` adds the `Dual`-stripping
# method, recursing so a higher-order tag chain still reduces to a scalar.
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
"
function solve_moment(::Type{D}, ::Val{names}, residual::R, deriv::G,
        bracket::B, vals) where {D, names, R, G, B}
    pvals = map(_primal, vals)
    lo, hi = bracket(pvals)
    f = s -> residual(s, pvals)
    _check_bracket(D, Val(names), vals, f(lo), f(hi))
    # AD-safety invariant: the Enzyme/Mooncake extensions hold this call
    # out of differentiation entirely (`EnzymeRules.inactive` /
    # `Mooncake.@zero_derivative` on `_solve_moment_equation`), which is
    # only correct because the loop below always runs immediately after
    # and reinjects the derivative from `vals`. Moving the solve or the
    # correction apart would silently return a zero (or garbage)
    # gradient on those two backends; see the extensions' own comments.
    s = _solve_moment_equation(f, lo, hi)
    for _ in 1:2
        s = s - residual(s, vals) / deriv(s, vals)
    end
    _check_solved(D, Val(names), residual, s, vals)
    return s
end

@doc raw"

Solve a family's own moment equations for its native parameters, exactly
and differentiably.

The vector counterpart of [`solve_moment`](@ref): where that inverts one
scalar equation with a supplied derivative and bracket, this inverts `N`
equations in `N` unknowns with no derivative supplied, so a family (or a
set of constraints other than the moments) only has to say what its
equations are and where to start.

- `residual(z, vals)` is the system, an `NTuple{N}` that is zero at the
  solution, evaluated at the unconstrained coordinates `z`.
- `seeds(pvals) -> Tuple` is one or more starting points, each an
  `NTuple{N}`, tried in order until one converges.

The iteration itself runs on `vals` stripped to its primal type, and the
same two implicit-function-theorem correction steps as `solve_moment`
recover the derivative in the caller's own type afterwards. The Jacobian
is held fixed at its finite-difference value through both: a step
`z - J \\ r` cancels whatever derivative `z` arrived with once `J` is the
Jacobian at the root, and the second step cancels the error in `J` itself
to second order, so both the gradient and the Hessian come back to
machine precision even though `J` never carries a derivative of its own.

Supports `N` of 1 and 2; a larger system raises.

# Arguments
- the native family being converted to.
- `Val(names)`: the alternative parameter names.
- `residual`, `seeds`: the system and its starting points.
- `vals`: the alternative parameter values, in `names` order.

# Examples
`solve_moments` is public but not exported, so import it by name.

```@example
using ReparameterisedDistributions, Distributions
using ReparameterisedDistributions: solve_moments

# A toy system, exp(z) = vals, solved for illustration; a real family's
# residual is its own moment equations.
solve_moments(Gamma, Val((:shape,)), (z, vals) -> (exp(z[1]) - vals[1],),
    pvals -> ((zero(pvals[1]),),), (2.0,))
```

# See also
- [`solve_moment`](@ref): the scalar counterpart.
- [`native_domains`](@ref): the transform the standard-moment fallback
  solves each native parameter under.
"
function solve_moments(::Type{D}, ::Val{names}, residual::R, seeds::S,
        vals) where {D, names, R, S}
    pvals = map(_primal, vals)
    # AD-safety invariant: the Enzyme/Mooncake extensions hold this call
    # out of differentiation entirely, exactly as they do the scalar
    # `_solve_moment_equation`, which is only correct because the
    # correction below always runs immediately after and reinjects the
    # derivative from `vals`.
    z, jac, converged = _solve_moment_system(w -> residual(w, pvals),
        seeds(pvals))
    _check_converged(D, Val(names), vals, converged)
    for _ in 1:2
        z = _corrected(z, jac, residual(z, vals))
    end
    _check_solved(D, Val(names), residual, z, vals)
    return z
end

# --- The damped Newton the vector solve runs on ------------------------------
#
# Written here rather than taken from a solver package: it is short enough
# that shipping it costs no dependency at all, which is what the
# `_solve_moment_equation` seam below exists to avoid. The Jacobian is
# central differences rather than the ForwardDiff seam for the same reason
# — the fallback then works with nothing but Distributions loaded — and it
# only ever enters as a value, never as a derivative.

# Central differences, one column per unknown. `cbrt(eps(T))` is the step
# that balances truncation against round-off for a central difference, so
# the Jacobian carries a relative error of order `eps(T)^(2/3)`, about
# 4e-11 in `Float64`. That error does not reach the gradient: the second
# correction step in `solve_moments` cancels it to second order, measured
# against `Gamma`'s own closed form as agreement to 3e-16.
function _fd_jacobian(f::F, z::NTuple{N, T}) where {F, N, T}
    h = cbrt(eps(T))
    return ntuple(Val(N)) do j
        hj = h * max(one(T), abs(z[j]))
        zp = ntuple(i -> i == j ? z[i] + hj : z[i], Val(N))
        zm = ntuple(i -> i == j ? z[i] - hj : z[i], Val(N))
        rp = f(zp)
        rm = f(zm)
        ntuple(i -> (rp[i] - rm[i]) / (2 * hj), Val(N))
    end
end

# `jac` is a tuple of COLUMNS, as `_fd_jacobian` builds it.
_linear_solve(jac::NTuple{1, <:NTuple{1, Real}}, r) = (r[1] / jac[1][1],)

function _linear_solve(jac::NTuple{2, <:NTuple{2, Real}}, r)
    a, c = jac[1]
    b, d = jac[2]
    det = a * d - b * c
    return ((d * r[1] - b * r[2]) / det, (a * r[2] - c * r[1]) / det)
end

function _linear_solve(jac::NTuple{N, <:NTuple{N, Real}}, r) where {N}
    throw(ArgumentError(
        "the numeric system solve handles one or two unknowns, not $(N)"))
end

function _corrected(z::NTuple{N}, jac, r) where {N}
    step = _linear_solve(jac, r)
    return ntuple(i -> z[i] - step[i], Val(N))
end

# Taking both operands as arguments rather than closing over them is what
# keeps the iteration inferred: a closure over a variable the loop below
# reassigns is boxed, and the whole solve comes back as `Tuple{Any, Any}`.
_step(z::NTuple{N}, λ, δ) where {N} = ntuple(i -> z[i] + λ * δ[i], Val(N))

_scaled(v::NTuple{N}, a) where {N} = ntuple(i -> a * v[i], Val(N))

_merit(r) = sum(abs2, r)

# Sum of squares for the line search, largest absolute residual for the
# convergence test: the merit has to fall on every accepted step, while
# convergence is a statement about each equation separately.
function _newton_solve(f::F, z0::NTuple{N, T}) where {F, N, T}
    z = z0
    jac = _fd_jacobian(f, z)
    for _ in 1:_NEWTON_ITERATIONS
        r = f(z)
        all(isfinite, r) || return (z, jac, false)
        maximum(abs, r) <= _newton_tol(T) && return (z, _fd_jacobian(f, z),
            true)
        jac = _fd_jacobian(f, z)
        all(c -> all(isfinite, c), jac) || return (z, jac, false)
        full = _linear_solve(jac, r)
        all(isfinite, full) || return (z, jac, false)
        # Cap the step in the unconstrained coordinates: an early Newton
        # step from a seed far from the root is otherwise large enough to
        # overflow `exp` and land on a NaN residual the line search can
        # only back away from one halving at a time.
        capped = min(one(T), T(_NEWTON_STEP_CAP) / max(maximum(abs, full),
            eps(T)))
        z, moved = _line_search(f, z, _scaled(full, -capped), _merit(r))
        moved || break
    end
    return (z, jac, maximum(abs, f(z)) <= _newton_tol(T))
end

function _line_search(f::F, z::NTuple{N, T}, δ, m0) where {F, N, T}
    λ = one(T)
    while λ > _LINE_SEARCH_MIN
        zt = _step(z, λ, δ)
        rt = f(zt)
        # Armijo: a step is accepted only on a decrease proportional to
        # its own length, so a shrinking sequence of near-zero
        # improvements cannot pass for convergence.
        if all(isfinite, rt) && _merit(rt) <= (1 - T(1.0e-4) * λ) * m0
            return (zt, true)
        end
        λ /= 2
    end
    return (z, false)
end

const _NEWTON_ITERATIONS = 100
const _NEWTON_STEP_CAP = 8.0
const _LINE_SEARCH_MIN = 1.0e-8

# Tighter than `_moment_atol`, which is what the corrected root is finally
# held to: the primal iterate is polished twice more before that check.
_newton_tol(::Type{T}) where {T} = eps(T)^(3 // 4)

# Runs the seeds in order and stops at the first that converges. Held out
# of differentiation by the Enzyme and Mooncake extensions (see
# `solve_moments`), so the branches, the line search and the finite
# differences inside it never reach a tape.
function _solve_moment_system(f::F, seeds::Tuple) where {F}
    z, jac, converged = _newton_solve(f, seeds[1])
    for k in 2:length(seeds)
        converged && break
        z, jac, converged = _newton_solve(f, seeds[k])
    end
    return (z, jac, converged)
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
# `valid_moments` reporting `false` before the solve runs — the same
# guard-first contract as an analytical family. (b) and (c) are below.

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

# (b'), the vector counterpart of the bracket check: no starting point
# reached the root, so there is no solution to correct. A family reached
# through the standard-moment fallback answers this in `valid_moments`
# first, by running the same iteration, so this is what an out-of-window
# request meets only when the guard was bypassed with `check_args = false`
# at construction and then asked for a moment or a quantile.
function _check_converged(::Type{D}, ::Val{names}, vals,
        converged::Bool) where {D, names}
    converged && return nothing
    throw(DomainError(vals,
        "the moment equations for $(D) by $(collect(names)) did not " *
        "converge from any starting point, so these moments are " *
        "outside the numerically solvable region for $(eltype(vals)); " *
        "if a native parameter of $(D) is not positive, register " *
        "`native_domains` for it"))
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

function _check_solved(::Type{D}, ::Val{names}, residual::R, z::Tuple,
        vals) where {D, names, R}
    r = maximum(abs, residual(z, vals))
    r <= _moment_atol(float(typeof(r))) && return nothing
    throw(DomainError(vals,
        "the numeric conversion of $(D) by $(collect(names)) did not " *
        "converge: largest residual $(r) at solved values " *
        "$(z), tolerance $(_moment_atol(float(typeof(r))))"))
end
