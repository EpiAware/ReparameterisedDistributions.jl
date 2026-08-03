# ComposedDistributions leaf-protocol hooks for a `Reparameterised` wrapper.
#
# A `Reparameterised` leaf's parameters are its registered moments, not the
# native family's own parameters (see src/Reparameterised.jl). `param_names`
# and `leaf_ctor` are the two hooks ComposedDistributions reads/rebuilds a
# leaf through, so together they fix `params_table`, `uncertain`,
# `build_priors` and `update` to work in moment coordinates rather than
# silently falling back to positional `:param_1`/`:param_2` labels.
#
# `free_leaf` is deliberately NOT overridden here: the default
# `free_leaf(leaf) = leaf` is already right, since a `Reparameterised` is its
# own free leaf. Overriding it to peel to `native(d)` would look like
# ordinary wrapper-transparency but would defeat the whole point of this
# package: `leaf_param_names`, `params_table` and `leaf_ctor` all peel through
# `free_leaf` before reading/rebuilding, so peeling to the native distribution
# would report and reconstruct in shape/scale coordinates instead of the
# registered moments. The other leaf-wrapper hooks (`uncertain_specs`,
# `extra_leaf_params`, `set_extra_leaf_params`, `leaf_mean`, `leaf_var`,
# `shared_tag`, `leaf_detail_lines`) are the leaf-*wrapper* half of the
# protocol; `Reparameterised` is a leaf, not a wrapper, so their defaults
# (reaching this leaf's own `mean`/`var`/`show`) are already correct and are
# left unoverridden.
#
# The flat codec (`flat_dimension`/`flatten`/`unflatten`/`reconstruct`) does
# NOT yet round-trip in moment coordinates: it reads the type-level
# `_params_arity_of`/`_param_names_of` companions instead of `param_names`,
# and those are not extensible from outside ComposedDistributions (a
# `@generated`-function world-age constraint; see the codec assertions in
# test/composed.jl, written `@test_broken`, and the upstream issue they
# point at). Land this extension's two working hooks anyway: `params_table`,
# `uncertain` and `update` already work in moment coordinates without the
# codec.
#
# A pre-existing, non-fixable wart, noted here rather than worked around:
# `_walk_rows!` gives every native-parameter row the leaf's own support
# (`(minimum(inner), maximum(inner))`), so a `mean` row and an `sd` row both
# get `(0.0, Inf)` for a LogNormal leaf. This is a generic behaviour that
# predates this extension (a plain `Normal`'s `sigma` row gets the same
# `(-Inf, Inf)` treatment), harmless for the parameterisations registered
# today.
module ReparameterisedDistributionsComposedDistributionsExt

using ComposedDistributions: ComposedDistributions
using ReparameterisedDistributions: Reparameterised, _build

# The estimable parameter names are the registered `names`, matched
# positionally to `Distributions.params(d) == d.vals`.
function ComposedDistributions.param_names(
        ::Reparameterised{D, names}) where {D, names}
    return names
end

# The callable `leaf_ctor` returns. A singleton parametric struct (rather than
# a closure) so that two structurally identical leaves return `===`
# constructors: `_tie_signature` groups tied leaves by this value, and a
# closure over a runtime value would make `tie`/`shared` wrongly reject two
# compatible leaves. `check_args` is accepted (and forwarded) even though
# nothing calls the ctor with it today, so `_ctor_has_check_args`'s dormant
# reflection answers `true` if that path is ever woken.
struct ReparameterisedLeafCtor{D, names} <: Function end

function (::ReparameterisedLeafCtor{D, names})(vals::Real...;
        check_args::Bool = true) where {D, names}
    return _build(D, Val(names), vals; check_args = check_args)
end

# Routes through the internal `_build`, not the public `reparameterise`,
# because the values arrive positionally (in `param_names` order) here,
# whereas `reparameterise` takes its names as keywords. `_build` is the
# shared construction path `reparameterise` itself uses, so canonicalisation
# and promotion match exactly.
function ComposedDistributions.leaf_ctor(
        ::Reparameterised{D, names}) where {D, names}
    return ReparameterisedLeafCtor{D, names}()
end

end
