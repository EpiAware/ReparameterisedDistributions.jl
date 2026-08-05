## Unreleased

### `Beta(mean, sd)` and `InverseGaussian(mean, sd)`: two more analytical moments

Registers `Beta(mean, sd)` (and `(mean, var)`), the natural coordinates for a
probability-scale quantity elicited as a central value and an uncertainty — a
reporting fraction or a case-fatality ratio, say — rather than the native
shape pair. Writing `nu = mean * (1 - mean) / var - 1` for the concentration,
`alpha = mean * nu` and `beta = (1 - mean) * nu` invert exactly. A Beta's
variance is bounded above by `mean * (1 - mean)`, the variance of a Bernoulli
with the same mean; a standard deviation elicited too wide for its mean has
no Beta at all, and is rejected by the validity guard rather than silently
clipped to the boundary.

Also registers `InverseGaussian(mean, sd)` (and `(mean, var)`). The native
`InverseGaussian(mu, lambda)` already takes the mean directly, so only the
shape needs deriving: `lambda = mean^3 / var`. The family is a
first-passage-time distribution (the hitting time of a drifting Wiener
process), a genuine alternative to the Gamma and log-normal for a
right-skewed delay such as an incubation period.

Both closed forms keep the package's contract: exact algebra, no solver, and
an explicit domain of validity rather than a silently nonsense distribution.
`Weibull(mean, sd)` was considered and deferred (#39): its CV-to-shape
relation has no elementary inverse, so it belongs to the numeric-fallback
seam tracked separately (#41), not here.

### `Weibull(mean, sd)`: the first numerically-solved parameterisation

Registers `Weibull` by `(mean, sd)` (and `(mean, var)`, delegating to it).
Unlike every other family, this has no exact closed form: the coefficient
of variation depends on the native shape alone, through a strictly
monotone one-dimensional equation, and the scale then follows in closed
form once the shape is known. The shape is found by a bracketing
root-find rather than exact algebra — a change to the package's own
architecture, not merely a new family:

- A family with no exact closed form registers exactly the same single
  `to_native` method as an analytic family, guarding its own validity
  first, and calls the new public `solve_moment(D, Val(names), residual,
  deriv, bracket, vals)` from inside its body once that guard passes.
  There is no separate trait or registry: a family's own `to_native`
  method is always more specific than the generic "no reparameterisation
  is registered" fallback, so ordinary dispatch reaches it whenever one is
  registered, and registering a closed form later for a currently-numeric
  pair is a plain method redefinition.
- The actual root-find lives in a new package extension,
  `ReparameterisedDistributionsRootsExt` (Roots.jl), kept out of the core
  package per the maintainer's ruling that this package owns the equation
  and the derivative rule but not a solver. Distributions.jl itself
  depends on Roots, so the extension loads for essentially every user
  without a new install.
- The gradient with respect to the moments stays exact regardless of how
  the root itself was found: two steps of an implicit-function-theorem
  correction recover the derivative afterwards from the residual equation
  alone, so a solver that returns a bare value (no derivative information
  at all) still yields the correct gradient and Hessian. This is verified
  directly against the classic failure mode — a solver silently truncating
  a tracked value to its primal, which would otherwise produce a *wrong*,
  often-zero gradient with no error — and against central differences.
- A second package extension, `ReparameterisedDistributionsForwardDiffExt`,
  strips a `ForwardDiff.Dual`-valued bracket to its primal before the
  solve. This is a robustness fix, not a correctness one — the correction
  above supplies the exact derivative either way — but it matters in
  practice: `Roots.A42` is not always reliable in `Dual` arithmetic
  (measured directly against this package's own equation, surfaced by an
  end-to-end Turing/NUTS run rather than a synthetic case).
  Enzyme and Mooncake are not yet given the same treatment; both currently
  fail (loudly, not silently) when differentiating the numeric
  conversion, tracing into Roots' own internals, and are recorded as
  known-broken in the AD test registry pending the same fix.
- Requesting a `(mean, sd)` outside the numerically solvable CV window —
  roughly `2.6e-4` to `3e23` in Float64, far wider than any
  epidemiologically meaningful delay — raises a `DomainError` under
  `check_args = true`, or gives `logpdf == -Inf`
  (`pdf == 0`) under `check_args = false`, exactly as for an
  analytically-invalid moment. A solve that runs but does not converge to
  within tolerance raises rather than ever returning a distribution whose
  moments differ from what was asked for.
- `loglikelihood(d::Reparameterised, x::AbstractArray)` now converts once
  and delegates to the native distribution's own batch method, rather
  than reconverting once per observation through Distributions.jl's
  generic scalar-`logpdf` summation. This benefits every family, but
  matters most for a numeric one, where each reconversion re-runs a
  root-find.

### Breaking: a single registration hook, `to_native`

**`to_native(D, Val(names), vals)` now returns `Union{Nothing, D}`, not
always `D`.** `nothing` means the parameters describe no member of the
family. This is a breaking change to an exported function's return type,
released as 0.2.0 rather than a patch, per SemVer 0.x.

Registering a family — analytic or numeric — needs exactly this one public
method. `valid_moments`, briefly public, is deleted, along with the
`_valid_moments` compat alias promised to keep working for one release: the
promise is broken a release early rather than shipped and immediately
superseded. A family's guard moves into the start of its own `to_native`
method, checked before any other work — in particular before a `sqrt` or
similar that would otherwise throw on invalid input instead of reporting it.

`native(d)` is unaffected in its own return type: it still unwraps to the
concrete native distribution and throws a `DomainError` on `nothing`, so
`@inferred native(d)` keeps passing and the allocation and timing profile is
unchanged. What is lost is the diagnostic of asking `native` for the
distribution an *invalid* wrapper would have converted to — checking that
now needs `to_native` directly, and reads `=== nothing` rather than a
distribution.

`solve_moment(D, Val(names), residual, deriv, bracket, vals)`, unaffected,
remains the numeric driver a family with no exact closed form calls from
inside its own `to_native` method, once its own guard has passed. This
registers no method on anything of this package's own: a numeric family's
registration footprint is identical in shape to an analytic one, one
`to_native` method.

Also fixed: `loglikelihood(d::Reparameterised, x::AbstractArray)` did not
guard at all, so a batch of observations against an invalid wrapper (built
with `check_args = false`) silently bypassed the whole validity contract —
the scalar `logpdf` path returned `-Inf`, the batched path did not. The
single hook makes this gap impossible to leave open: `nothing` cannot be
passed to `Distributions.loglikelihood`, so the branch has to be written.

Deleted outright, from the two-hook design above: the `_conversion_kind`
trait, the `Analytic`/`Numeric` marker types, and the four private hooks a
numeric family previously had to add methods to (`_moment_residual`,
`_moment_residual_deriv`, `_moment_bracket`, `_from_solution`). None of them
were an irreducible extension point — a family's own `to_native` method was
already strictly more specific than the generic fallback, so it was always
chosen by ordinary dispatch regardless of the trait.

### `rescale`: scale a registered moment while holding the others fixed

`rescale(d, factor; parameter = :mean)` scales `d`'s named registered
parameter by `factor`, routing through whichever moment parameterisation `d`
was itself built under rather than requiring a caller to look up the family's
registered names and rebuild the wrapper by hand. An affine transform is not a
substitute for a discrete family such as `NegativeBinomial`, where scaling the
native support does not scale the mean cleanly; `rescale` scales in moment
coordinates and reconverts through the family's own closed form. Naming a
parameter the family is not registered under raises a `DomainError`; calling
`rescale` on a native, unwrapped `Distribution` raises an `ArgumentError`
naming the family to wrap first.

### `NegativeBinomial(mean, dispersion)`: the reciprocal overdispersion convention

Registers a second `NegativeBinomial` parameterisation, `(mean, dispersion)`
with `var = mean + mean^2 / dispersion`, alongside the existing
`(mean, overdispersion)` with `var = mean + overdispersion * mean^2`. The two
are reciprocals (`dispersion = 1 / overdispersion`); previously a caller
supplying a dispersion value under the `overdispersion` name (or vice versa)
silently obtained a distribution with the wrong variance, which is exactly the
confusion the wrapper exists to remove.

### Rate parameterisations for `Exponential` and `Gamma`

Registers `Exponential(rate)` and `Gamma(shape, rate)`, so a distribution
written in terms of a hazard rate — the quantity a prior is usually placed on,
and the quantity one wants reported — no longer has to be hand-inverted to a
scale before it can be composed. Both invert exactly: `scale = 1 / rate`. A
non-positive rate is rejected by the validity guard rather than silently
building an invalid native distribution.

### `SkewNormal(centre, scale, mass_below_centre)`: a tail-probability parameterisation

Registers a fourth family, `SkewNormal`, keyed not on a moment but on the
probability mass falling below a reference point — a recurring elicitation
form. For the untruncated family that mass depends only on the native shape
and inverts exactly (`alpha = tan(pi * (1/2 - mass_below_centre))`), so the
addition keeps the package's contract of exact, differentiable, solver-free
algebra. The elicited fraction holds exactly only for the untruncated
distribution; truncating the result makes it approximate.
Distributions.jl does not implement `cdf`/`quantile` for `SkewNormal` (Owen's
T function is not implemented there), a limitation this parameterisation
inherits rather than works around.

### The moment-parameterised wrapper

`reparameterise(dist_or_type; moments...)` wraps a Distributions.jl family so
that the quantities a modeller reasons about are its parameters. The result is an
ordinary `Distribution`: it evaluates and samples exactly as the native one does
and goes on the right of a `~`, so a model puts priors directly on a mean and a
standard deviation and samples in those coordinates. `params` reports the
moments, not the native parameters that only imply them.

The conversion to the native family is exact algebra rather than a numerical
solve, so it is differentiable; gradients with respect to the moments are checked
against ForwardDiff, ReverseDiff, Enzyme (forward and reverse) and Mooncake
(forward and reverse).

Supported: `LogNormal` by `(mean, sd)` or `(mean, var)`; `Gamma` by `(mean, sd)`,
`(mean, var)` or `(mean, shape)`; and `NegativeBinomial` by
`(mean, overdispersion)`, where the overdispersion is the excess variance
relative to a Poisson (`var = mean + overdispersion * mean^2`).

A wrapper takes its variate form and value support from the family it wraps, so a
`NegativeBinomial` wrapper stays discrete.

The moments are validated in their own coordinates, not merely through the native
distribution they imply. The LogNormal and Gamma conversions square the standard
deviation, so a negative one would otherwise map onto a perfectly valid native
distribution and the wrapper would report a parameter it does not behave as.

### `native` and `to_native` are now exported

The wrapper-level accessor and the per-family extension point were both
previously underscore-prefixed internals (`_native`, `_to_native`), against
the org convention that a name a caller is expected to type directly is
exported rather than merely public. `native(d)` reaches the distribution a
wrapper converts to — and, through it, the native parameters
(`params(native(d))`) when the moments alone are not what is needed.
`to_native(D, Val(names), vals)` is unchanged in behaviour; only its name and
visibility change, so registering a new family from outside this package no
longer needs the `ReparameterisedDistributions.` qualification.

### Construction is now fully type-inferred

`reparameterise` returned a `Reparameterised` with uninferred `names`, `N`
and `T` type parameters for any call the compiler did not fully
constant-fold — `@inferred` on it failed. The cause: the internal `_build`
took the parameter names as a plain `Tuple{Vararg{Symbol}}` and constructed
a `Val` from it internally, and `Val(runtime_value)` cannot be inferred
concretely from a value the caller has not already carried as a type
parameter. `_build` now takes `Val{names}` directly, so the whole
`Reparameterised{...}` type is inferred end to end — the case that matters
most is exactly the one constant folding cannot be relied on inside: an AD
tape.

### Package identity

The package is renamed from `AltDistributions` to `ReparameterisedDistributions`
to match the repository, and takes a new UUID
(`7cd6e41d-e337-45a7-b8fc-acb99a44bf42`).
Neither the old name nor the old UUID could be carried forward: the 2024
scaffold's UUID is now held in the General registry by `CensoredDistributions`,
which was derived from that scaffold, and the name `AltDistributions` is held
there by an unrelated package.

### Polish

A `Reparameterised` now has a richer REPL display: alongside the
code-reconstructable one-liner, `MIME"text/plain"` also prints the native
distribution the wrapper actually evaluates as.
The package gains its own logo, the getting-started page has a runnable
first example rather than scaffold placeholder text, and a stray reference
to an unpublished internal package has been removed from the docs.

### Infrastructure

The package adopts the EpiAwarePackageTools managed standard with `ad = true`:
managed CI, quality checks (Aqua, ExplicitImports, JET, docstring format,
formatting), the Documenter and DocumenterVitepress docs build, a benchmark
suite, and the AD-gradient harness covering ForwardDiff, ReverseDiff, Enzyme
(forward and reverse) and Mooncake (forward and reverse).
