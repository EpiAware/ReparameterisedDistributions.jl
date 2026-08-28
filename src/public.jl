# Public API declarations for Julia 1.11+ (public but not exported).

# The wrapper type and its supertype. `reparameterise` is the exported verb that
# builds them; the types themselves are public so a downstream package can
# dispatch on them, per the ecosystem convention that verbs are exported and
# types are public.
public AbstractReparameterisedDistribution, Reparameterised

# `solve_moment` and `solve_moments` are the drivers a family with no exact
# closed form calls from within its own `to_native` method — one scalar
# equation, or a system of them — and `test_reparameterisation` is the
# interface test suite a registration — this package's own or another
# package's — is checked against. All are called rather than extended, so
# all are public rather than exported, like the registration hooks.
public solve_moment, solve_moments, test_reparameterisation

# `native_domains` is the third extension point, read by the
# standard-moment fallback: a family whose native parameters are not all
# positive registers it so the fallback solves in the right coordinates.
public native_domains

# `to_native` and `valid_moments` are the two per-family extension points a
# registration implements (see families.jl for the pattern a new
# registration follows). A registration author extends them by name, so
# `using ReparameterisedDistributions` alone is not enough to reach them —
# they must be named explicitly, either through `import` or a qualified
# method definition (see the developer guide). That makes them public
# rather than exported: unlike `reparameterise`, `rescale` and `native`,
# these are not names an ordinary caller types directly.
public to_native, valid_moments
