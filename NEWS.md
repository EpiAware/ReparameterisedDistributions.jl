## Unreleased

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
first example rather than scaffold placeholder text, and stray references
to packages and names outside this repository's own attribution have been
removed.

### Infrastructure

The package adopts the EpiAwarePackageTools managed standard with `ad = true`:
managed CI, quality checks (Aqua, ExplicitImports, JET, docstring format,
formatting), the Documenter and DocumenterVitepress docs build, a benchmark
suite, and the AD-gradient harness covering ForwardDiff, ReverseDiff, Enzyme
(forward and reverse) and Mooncake (forward and reverse).
