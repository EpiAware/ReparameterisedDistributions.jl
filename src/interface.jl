@doc raw"

Supertype for a distribution that stands in for a native Distributions.jl
family under a different parameterisation.

A subtype stores the alternative parameters and converts to the native family
on demand, so it evaluates exactly as the native distribution does while
reporting the alternative parameters as its own. The variate form and value
support are carried as type parameters and taken from the family being wrapped,
so a wrapper around a discrete family stays discrete.

# See also
- [`reparameterise`](@ref): the public constructor.
- [`Reparameterised`](@ref): the concrete wrapper.
"
abstract type AbstractReparameterisedDistribution{F <: VariateForm,
    S <: ValueSupport} <: Distribution{F, S} end
