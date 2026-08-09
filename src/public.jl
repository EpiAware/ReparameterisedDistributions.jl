# Public API declarations for Julia 1.11+ (public but not exported).

# The wrapper type and its supertype. `reparameterise` is the exported verb that
# builds them; the types themselves are public so a downstream package can
# dispatch on them, per the ecosystem convention that verbs are exported and
# types are public.
public AbstractReparameterisedDistribution, Reparameterised

# `solve_moment` is the driver a family with no exact closed form calls from
# within its own `to_native` method, and `test_reparameterisation` is the
# interface test suite a registration — this package's own or another
# package's — is checked against. Both are called rather than extended, so
# both are public rather than exported, unlike the two registration hooks.
public solve_moment, test_reparameterisation

# `to_native` and `valid_moments` (the two per-family extension points — see
# families.jl for the pattern a new registration follows) and `native` (the
# wrapper-level accessor) are exported instead of merely public — see the
# main module file — because, unlike this package's other internals, a
# caller is expected to type these names directly rather than dispatch on a
# type.
