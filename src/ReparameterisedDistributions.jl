"""
    ReparameterisedDistributions

Parameter-convention switches for Distributions.jl: wrap a distribution so that
it is parameterised by the quantities a modeller reasons about — its moments —
rather than by its native parameters.

This release ships the package's managed infrastructure only. The
`reparameterise` wrapper and its closed-form conversions land next (#19, #20).

# Examples
```@example
using ReparameterisedDistributions
```
"""
module ReparameterisedDistributions

# Docstring-template machinery used by src/docstrings.jl (imports are
# centralised here per the kit's import-centralisation gate).
using DocStringExtensions: @template, DOCSTRING, EXPORTS, IMPORTS, TYPEDEF,
                           TYPEDFIELDS, TYPEDSIGNATURES

# Register the standard EpiAware docstring conventions before any docstrings
# are defined (see src/docstrings.jl).
include("docstrings.jl")

end # module ReparameterisedDistributions
