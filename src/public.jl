# Public API declarations for Julia 1.11+ (public but not exported).

# The wrapper type and its supertype. `reparameterise` is the exported verb that
# builds them; the types themselves are public so a downstream package can
# dispatch on them, per the ecosystem convention that verbs are exported and
# types are public.
public AbstractReparameterisedDistribution, Reparameterised

# The extension point for a new family: add a `_to_native` method for the
# (family, parameter-name) pair. It is public because registering a family from
# outside this package is a supported use, not an internal detail.
public _to_native
