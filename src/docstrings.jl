# Standard EpiAware docstring conventions (recreates CensoredDistributions.jl
# `src/docstrings.jl`). DocStringExtensions `@template` blocks give every
# function, type, and the module a consistent docstring layout: a signature
# header, the authored prose, and — for types — an auto-generated field list.
#
# PACKAGE-OWNED: scaffold writes this once and never overwrites it. To activate
# it, `include` this file near the TOP of the package module, AFTER the
# module's `using DocStringExtensions: ...` (all genuine module-scope
# `using`/`import` statements live in the main module file, not here) and
# BEFORE any docstrings are defined (a `@template` only applies to
# docstrings written after it in the same module):
#
#     module MyPackage
#     using DocStringExtensions: @template, DOCSTRING, EXPORTS, IMPORTS,
#                                TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
#     include("docstrings.jl")   # registers the @template conventions
#     # ... the rest of the package, with docstrings ...
#     end
#
# Add DocStringExtensions to the package `[deps]` (this file's `@template`
# blocks need it, imported by the module's own `using` above).
# `scaffold_generate` wires both for a fresh package automatically. It pairs with
# `test_docstring_format` (which checks the rendered docstrings) and the
# Documenter + DocumenterVitepress build in `docs/make.jl`.

@template (FUNCTIONS, METHODS, MACROS) = """
                                         $(TYPEDSIGNATURES)
                                         $(DOCSTRING)
                                         """

@template TYPES = """
                  $(TYPEDEF)
                  $(DOCSTRING)

                  ---
                  ## Fields
                  $(TYPEDFIELDS)
                  """

@template MODULES = """
                    $(DOCSTRING)

                    ---
                    ## Exports
                    $(EXPORTS)
                    ---
                    ## Imports
                    $(IMPORTS)
                    """
