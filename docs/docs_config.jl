# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Package-specific configuration read by the managed `make.jl`. It drives the
# Literate.jl tutorial pipeline and the README/index link rewrites, and lists
# the linkcheck URLs to ignore. The defaults below build a site with no
# tutorials, so a fresh package needs no edits here; fill these in as the docs
# grow. CensoredDistributions.jl's `docs/make.jl` is a worked example of the
# values these consts take.

# Tutorial source `.jl` files (Literate scripts) under `TUTORIALS_SUBDIR`.
#
# Light tutorials emit `@example` blocks that Documenter runs in-process; keep
# cheap tutorials here.
const LIGHT_TUTORIALS = String[]

# Heavy tutorials (live MCMC fits, multi-backend AD, plotting) are each
# executed once in a fresh subprocess so native/memory state cannot accumulate.
# The `ad-backends.jl` entry is seeded when the package is scaffolded with
# `ad = true`: the page itself is kit-managed (re-applied on every sync); only
# this registration is package-owned.
const HEAVY_TUTORIALS = String[
    "ad-backends.jl"
]

# Where the tutorial `.jl` sources and rendered `.md` pages live, relative to
# `docs/src`.
const TUTORIALS_SUBDIR = joinpath("getting-started", "tutorials")

# Fast-build stubs (`--skip-notebooks`): `"file.md" => "# Heading"` pairs. The
# heading should preserve the tutorial's `@id` (e.g.
# `"# [Title](@id my-anchor)"`) so cross-references from other pages still
# resolve in a fast build.
const TUTORIAL_STUBS = Pair{String, String}[
    "ad-backends.md" => "# [Automatic differentiation backends](@id ad-backends)"
]

# Heavy tutorials that always render from their `TUTORIAL_STUBS` heading and
# never execute, independent of `--skip-notebooks` — the escape hatch for a
# heavy tutorial with a problem of its own (e.g. a model that does not
# terminate in reasonable time), so it need not block its siblings from
# running for real. Leave empty; every heavy tutorial with no such problem
# should execute.
const FORCE_STUB_TUTORIALS = String[]

# Regexes for URLs to skip during the (full-build) linkcheck, e.g. a page
# published by a separate workflow that is not yet live.
#
# The stable docs URL 404s until the first tagged release deploys, and the
# README badge row links it, so the deploy build's linkcheck (a hard error, not
# a warning) fails without this. Drop the ignore once /stable is live; /dev
# already resolves.
#
# The managed README also links `CITATION.cff` at `blob/main/`, and that file
# only reaches `main` when the adoption itself merges — so on the adopting branch
# the link 404s and fails the build that has to pass before the merge can happen.
# Drop this second ignore once the adoption has landed on `main`.
const LINKCHECK_IGNORE = [
    r"^https://epiaware\.org/ReparameterisedDistributions\.jl/stable",
    r"^https://github\.com/EpiAware/ReparameterisedDistributions\.jl/blob/main/CITATION\.cff"
]

# README -> index.md link rewrites: `from => to` pairs applied line by line,
# e.g. rewriting an absolute docs URL to an in-site `@ref` so links stay within
# the built version.
const INDEX_REWRITES = Pair{String, String}[]

# Whether README ```julia blocks become runnable `@example readme` blocks on the
# generated home page. Keep `true` when the README's examples are real, runnable
# code; set `false` when they are illustrative (placeholder names) and must not
# execute.
#
# `false` here: the package is unregistered, so the README's only code block is
# the `Pkg.add(url = ...)` install snippet. Executing it during the docs build
# installs the package from `main`, which is still the 2024 `AltDistributions`
# scaffold, into the docs environment and renders that output onto the home
# page. Revisit once there are real runnable examples to show (#19, #20).
const README_EXECUTE = false

# README headings whose whole section (heading + body, up to the next heading
# of the same or a higher level) is dropped when generating the home page. The
# managed badge block is always stripped via its `<!-- badges:start/end -->`
# markers; this list is the package-owned hook for omitting any OTHER named
# section from the home page (the managed build hardcodes none). Leave empty to
# keep the whole README — content tables and all.
const INDEX_STRIP_SECTIONS = String[]

# Whether the build generates the benchmark page (`src/benchmarks.md`): the
# package-owned `docs/benchmarks.md` prose hook plus an overall summary
# table + combined trend plot and the per-suite detail, both rendered from
# the timeline published to the repo's `benchmarks` branch. Defaults to the
# `benchmarks` flag the package was scaffolded with; `false` drops the page
# and `make.jl` also omits its `pages.jl` nav entry. The trend plot needs
# `Plots` in `docs/Project.toml` (lazily loaded, so it degrades to a
# table-only page with an `@info` note when absent rather than failing the
# build).
const BENCHMARK_PAGE = true

# Headline benchmark suites to keep on the performance-history page. A suite is
# the first `/`-segment of a benchmark's name (e.g. "AD gradients" in
# "AD gradients/Convolved Normal+Normal/ForwardDiff"). Empty keeps every suite;
# name a few here when the full suite list makes the history page too long.
const HISTORY_SUITES = String[]

# How many of the most-recent revisions (columns) to show in the overall
# summary and history ratio table. The published `table.md` can carry every
# benchmarked release; this caps the rendered table (and trend plot) so it
# stays readable. Columns are relabelled with commit dates.
const HISTORY_COMMITS = 5

# The overall-summary ratio (a suite's median benchmark value at the most
# recent shown revision, against its value at the oldest shown revision) at
# or above which that suite's `Status` flags "⚠ reg". 1.1 == a 10% increase
# in runtime/memory counts as a regression; raise it for a noisier benchmark
# suite, lower it for a stricter one. Must be > 1.0 — at or below that, a
# suite with no change (ratio 1.0) or even an improvement would flag.
const HISTORY_REGRESSION_THRESHOLD = 1.1
