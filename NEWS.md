## Unreleased

### Fix: `to_native` splits back into `to_native` and `valid_moments` (#86)

`to_native` folded its validity guard into itself and returned
`Union{Nothing, D}` instead of a concrete `D` (#84). On x86_64, Enzyme
reverse mode computed a silently wrong gradient — no error, no warning —
for the two `NegativeBinomial` parameterisations, because their native
`logpdf` is a hand-written `Distributions.jl` method rather than one
delegated to StatsFuns; every other registered family, every other AD
backend, and Enzyme forward mode were unaffected.

The fix keeps `to_native` from ever binding a `Union{Nothing, D}`-typed
value on the differentiated hot path (`logpdf`, `pdf`, `loglikelihood`) by
splitting validity back out into its own hook, `valid_moments`, checked at
every call site before `to_native` runs. `to_native` itself no longer
guards: it always returns a concrete distribution, and a family registers
it on parameters already known valid.

This is a partial reversal of #84's "one hook" simplification (#80):
registering a family is two methods again, `valid_moments` and `to_native`,
not one. The two-hook shape is not new — it is the one #84 collapsed, with
the predicate now exported rather than internal, since a family registered
from another package has to be able to write it. `to_native`'s call
signature, and its
`nothing`-on-invalid behaviour from a caller's point of view (`native`
still throws a `DomainError`, `logpdf`/`pdf` still give `-Inf`), are
unchanged.

This is a breaking change to the registration contract. `v0.2.0` shipped
`to_native`'s `Union{Nothing, D}` return and is tagged and registered, so a
downstream package that registered a family against it has a `to_native`
method that guards its own input and returns `nothing`. Such a method now
needs its guard moved into a `valid_moments` method. Left unmoved it fails
loudly rather than silently — the 3-arg `valid_moments` fallback reports
the point valid, `to_native` returns `nothing`, and the density raises a
`MethodError` on `nothing` at exactly the invalid point that used to give
`-Inf` — and it puts the union back on the hot path for valid points too,
which is the shape this release exists to remove.

A regression guard in `test/families.jl` asserts every registered
`to_native` method's inferred return type is a concrete distribution and
never admits `nothing`. `@inferred` cannot do this job: union-splitting
narrows `logpdf`'s own return type back to a concrete `Float64` on every
branch, so it passed throughout the bug.
