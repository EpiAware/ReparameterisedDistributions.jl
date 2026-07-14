## Unreleased

### Package identity

The package is renamed from `AltDistributions` to `ReparameterisedDistributions`
to match the repository, and takes a new UUID
(`7cd6e41d-e337-45a7-b8fc-acb99a44bf42`).
Neither the old name nor the old UUID could be carried forward: the 2024
scaffold's UUID is now held in the General registry by `CensoredDistributions`,
which was derived from that scaffold, and the name `AltDistributions` is held
there by an unrelated package.

### Infrastructure

The package adopts the EpiAwarePackageTools managed standard with `ad = true`:
managed CI, quality checks (Aqua, ExplicitImports, JET, docstring format,
formatting), the Documenter and DocumenterVitepress docs build, a benchmark
suite, and the AD-gradient harness covering ForwardDiff, ReverseDiff, Enzyme
(forward and reverse) and Mooncake (forward and reverse).
