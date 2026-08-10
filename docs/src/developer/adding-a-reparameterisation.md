```@meta
CurrentModule = ReparameterisedDistributions
```

# [Adding a reparameterisation](@id adding-a-reparameterisation)

A parameterisation is registered by two methods.
There is no registry to append to, no trait to opt into and no macro to call.
Adding the two methods is the whole of it, whether they live in this package or in yours.

This page is the contract those two methods have to keep, a worked example that keeps it, and the test suite that checks a registration against it.

```@example adding
using ReparameterisedDistributions, Distributions
```

## The two hooks

| Hook | Signature | Returns | Runs |
|---|---|---|---|
| [`valid_moments`](@ref) | `(::Type{D}, ::Val{names}, vals)` | `Bool` | first, at every call site |
| [`to_native`](@ref) | `(::Type{D}, ::Val{names}, vals)` | a concrete `D` | only once the predicate has passed |

Both are exported, and both take the same three arguments.

`D` is the native Distributions.jl family being converted to.
`names` is the tuple of alternative parameter names, carried as a `Val` so the pair is resolved at compile time rather than by comparing symbols at run time.
`vals` is the alternative parameter values, in `names` order, already promoted to a common floating-point type.

Every call site in the package — `reparameterise`, `native`, `logpdf`, `pdf`, `loglikelihood` and the REPL `show` method — checks the predicate and then, separately, calls the conversion.
Nothing calls the conversion without checking first.

Both are exported rather than merely public, because a registration has to add methods to them.
Adding a method to a name brought into scope by `using` is an error in Julia, so import them explicitly or qualify the definitions:

```julia
import ReparameterisedDistributions: to_native, valid_moments
# or write `function ReparameterisedDistributions.to_native(...)` in full
```

## What `valid_moments` must guarantee

It answers whether the given values describe a member of the family, without converting and without throwing.

It has to decide this **in the alternative parameters' own coordinates**, because the native distribution's type cannot always recover the answer.
The `LogNormal` and `Gamma` conversions square the standard deviation, so a negative standard deviation builds exactly the same, perfectly valid native distribution as its positive counterpart.
By the time the conversion has run, the invalidity is gone.

It also has to cover combinations that are individually fine but jointly unattainable.
A `Beta`'s variance cannot exceed `mean * (1 - mean)`, so a positive mean in `(0, 1)` and a positive standard deviation can still describe no `Beta` at all.
For a numerically converted family it also has to state the window the root-find can actually solve, so an out-of-window request is a zero density rather than a throw from inside the solver.

Return `Bool`, not a number and not `nothing`: every differentiated call site branches on this value.

The three-argument fallback returns `true`, so a family that registers only a conversion is silently treated as always valid.
That is rarely what you want, and it is rarely loud: `check_args = true` construction still raises for many invalid points, but on the `check_args = false` hot path nothing surfaces the omission, and `logpdf`/`pdf`/`loglikelihood` return a finite, wrong density instead of `-Inf`.
Register both methods together.

## What `to_native` must guarantee

It performs the conversion, unconditionally, and returns a concretely-typed distribution of the family.

It must not guard its own input.
The predicate has already run at every call site, so the conversion may assume its input is valid and build whatever it builds.

It must not return `nothing`, and must not contain a branch whose two arms return different types.

Build the native distribution with `check_args = false`.
The conversion sits on the sampler's hot path and runs inside a gradient; the checks have already happened in moment coordinates, and re-running the family's own checks here would raise mid-gradient at exactly the invalid points that are meant to give `-Inf`.
Construction still forces one pass through the native family's own checks, once, so a moment that is individually valid but not representable (`mean = Inf`, say) is still caught up front.

### Why the concrete return type is a hard requirement

This is the one part of the contract that is not merely tidiness.

An earlier release folded the guard into the conversion and returned `Union{Nothing, D}`, with each call site branching on the result.
On x86_64, Enzyme reverse mode then computed a **silently wrong gradient** for the two `NegativeBinomial` parameterisations: no error, no warning, just wrong numbers.
Binding a union-typed value on the differentiated path was enough to trigger it.

Julia's own inference hides this from the obvious test.
Union-splitting narrows the *outer* return type of `logpdf` back to a concrete `Float64` on every branch, so `@inferred logpdf(d, x)` passed throughout the bug.
The instability is only visible one level in, on the conversion's own return type — which is what [`test_reparameterisation`](@ref) and the package's own regression guard check.

So: check validity first, convert second, and keep the conversion's return type concrete.

## Parameter names are sorted before dispatch

Keyword arguments are order-insensitive everywhere else in Julia, and `reparameterise` keeps them that way by sorting its keyword names alphabetically before dispatching.

Register both methods under the **sorted** names.
`Val((:mean, :sd))` is found; `Val((:sd, :mean))` is never reached.
For `NegativeBinomial` by dispersion and mean the canonical order is `(:dispersion, :mean)`, and for `Gamma` by shape and rate it is `(:rate, :shape)`, however the user types the keywords.

`params` reports the values in that same sorted order.

## A worked example

`Laplace` is not registered by this package.
A `Laplace(mu, theta)` has mean `mu` and variance `2 * theta^2`, so the location is native and `theta = sd / sqrt(2)`.
The location is unconstrained, so only the scale needs guarding.

The predicate first:

```@example adding
import ReparameterisedDistributions: to_native, valid_moments

function valid_moments(::Type{Laplace}, ::Val{(:mean, :sd)}, vals)
    _, sd = vals
    return sd > 0
end
```

Then the conversion, which assumes the predicate has passed:

```@example adding
function to_native(::Type{Laplace}, ::Val{(:mean, :sd)}, vals)
    mean, sd = vals
    return Laplace(mean, sd / sqrt(oftype(sd, 2)); check_args = false)
end
```

`oftype(sd, 2)` rather than a bare `2` keeps a `Float32` — or a dual number under automatic differentiation — from being widened by the division.

That is the whole registration.

```@example adding
d = reparameterise(Laplace; mean = 3.0, sd = 2.0)
```

```@example adding
(params(d), mean(d), std(d), logpdf(d, 2.5))
```

An invalid scale is rejected at construction, and gives a zero density without the construction check rather than raising inside a gradient:

```@example adding
bad = reparameterise(Laplace; mean = 3.0, sd = -2.0, check_args = false)

logpdf(bad, 2.5)
```

## Families with no closed form

A family whose conversion has no exact algebra registers exactly the same two methods.
Its `to_native` calls [`solve_moment`](@ref) inside its own body, passing its moment equation, that equation's derivative, and a bracket as ordinary functions.

```julia
function to_native(::Type{MyFamily}, ::Val{(:mean, :sd)}, vals)
    s = solve_moment(MyFamily, Val((:mean, :sd)), residual, deriv,
        bracket, vals)
    return MyFamily(...; check_args = false)
end
```

The root-find runs on the parameters stripped to their primal type, and the exact derivative is recovered afterwards by an implicit-function-theorem correction, so the conversion stays differentiable whatever the solver returns.
`valid_moments` carries an extra duty here: it must also exclude requests the bracket cannot answer, so the solver is never reached with them.

`Weibull` by mean and standard deviation is the worked example in the package itself (`src/families.jl`), and the solver backend is supplied by a package extension, so a numeric family needs `Roots` loaded.

## Testing a registration

[`test_reparameterisation`](@ref) checks a registered pair against the whole contract above in one call.
It is public but not exported, and `Test` supplies it through a package extension, so reach it with `using Test` and an explicit import.

```@example adding
using Test
using ReparameterisedDistributions: test_reparameterisation

test_reparameterisation(Laplace, (:mean, :sd), (3.0, 2.0);
    invalid = ((3.0, -2.0), (3.0, 0.0)))
```

It checks that:

- the pair is registered at all, rather than falling through to the error-raising fallback, and is registered under the canonical sorted names;
- `valid_moments` returns a `Bool`, and returns `true` at the point given;
- `to_native` is inferred to a single, concrete distribution type of the family, and never admits `nothing`, which is the invariant the section above exists for;
- the wrapper builds, reports the given values as its `params`, and keeps the family's variate form and value support, so a discrete family stays discrete;
- `native`, `logpdf` and `pdf` are type-inferred, and the densities and draws agree with the native distribution;
- any parameter named `mean`, `sd` or `var` comes back out of the built distribution, which is what checks the conversion's algebra rather than only its types;
- each `invalid` tuple is refused by the predicate, raises a `DomainError` at construction, and gives `logpdf == -Inf` and `pdf == 0` with `check_args = false`.

Pass at least one `invalid` tuple.
Without one, nothing checks that the guard guards, and a family that registered no predicate at all would pass.
`rtol` loosens the `mean`/`sd`/`var` round-trip for a numerically converted family.

### From your own package

Add `Test` to your test environment, and call the suite from your own testset.
A registration that needs a root-find also needs `Roots` there.

```julia
using Test, Distributions, ReparameterisedDistributions
using ReparameterisedDistributions: test_reparameterisation

@testset "MyPackage reparameterisations" begin
    test_reparameterisation(Laplace, (:mean, :sd), (3.0, 2.0);
        invalid = ((3.0, -2.0), (3.0, 0.0)))
end
```

The suite goes through the public surface only, so it holds your registration to exactly the contract this package holds its own to.
It is not a substitute for testing your conversion's algebra against values worked by hand, which is what catches a formula that is self-consistent but wrong.

## Contributing a family to this package

The same two methods, plus what a shipped family owes its users.

1. Add both methods to `src/families.jl`, next to the family they are closest to, with a comment deriving the algebra.
2. Add a case to the table of registered pairs in `test/interface.jl`, and update the count assertion beside it.
3. Add a testset to `test/families.jl` checking the conversion against native parameters worked out by hand.
4. Add a row to the table of supported parameterisations in the [getting started](@ref getting-started) page.

A numeric family also belongs in the AD scenarios under `test/ad/`, since a root-find is where a gradient is most likely to go wrong.
