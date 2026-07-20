# Public API declarations for Julia 1.11+ (public but not exported).

# The wrapper type and its supertype. `reparameterise` is the exported verb that
# builds them; the types themselves are public so a downstream package can
# dispatch on them, per the ecosystem convention that verbs are exported and
# types are public.
public AbstractReparameterisedDistribution, Reparameterised

# `to_native` (the per-family extension point) and `native` (the wrapper-level
# accessor) are exported instead of merely public — see the main module file
# — because, unlike this package's other internals, a caller is expected to
# type these names directly rather than dispatch on a type.
