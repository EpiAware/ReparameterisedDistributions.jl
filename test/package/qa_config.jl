# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# QA configuration values the managed `quality.jl` testset reads. Fill in the
# package-specific inputs the shared helpers need; the standard testset logic
# stays in `quality.jl` (managed). Edit freely.

using ReparameterisedDistributions

const QA_CONFIG = (
    # The module under test.
    mod = ReparameterisedDistributions,

    # Path to the isolated JET environment (see test/jet/Project.toml).
    jet_env = joinpath(@__DIR__, "..", "jet"),

    # Per-check Aqua relaxations, e.g. (; ambiguities = false). Empty = all on.
    aqua = (;),

    # ExplicitImports `ignore`: symbols the main module legitimately imports
    # non-publicly. Tuple of Symbols, e.g. (:_internal_helper,). Package
    # extensions are handled automatically (#189), so their import lists do not
    # need listing here.
    ei_ignore = (),

    # Docstring `crossref_ignore`: upstream names docstrings link to via
    # `[`name`](@ref)`, e.g. (:pdf, :cdf, :logpdf).
    crossref_ignore = (),

    # Extra docstring-format options, e.g.
    # (; exported_only_examples = true, require_field_docs = true).
    docstring = (;),

    # README section-structure check. `path` is the package root (its
    # README.md). Override `required`/`order` to extend or relax the standard
    # section set, e.g.
    #   (; required = vcat(STANDARD_README_SECTIONS, [("Benchmarks",)]))
    # Empty `(;)` uses the standard structure in standard order.
    readme = (; path = joinpath(@__DIR__, "..", "..")),

    # Package extensions to ambiguity-check. Each entry:
    #   (; name = :MyPkgSomeTriggerExt,
    #      triggers = ("SomeTrigger",),       # packages to load first
    #      prefixes = ("MyPkg", "SomeTrigger"),
    #      expect_phantoms = false,    # true if a third party adds phantoms
    #      broken = false)             # true to quarantine a known ambiguity
    extensions = ()
)
