# The generic standard-moment fallback (src/standard_moments.jl): a family
# with no registered closed form is converted by solving its own moment
# equations for its native parameters.

@testitem "an unregistered family converts from (mean, sd)" begin
    using Distributions

    # `Frechet` is not registered by this package, so this reaches the
    # generic numeric fallback rather than any closed form.
    d = reparameterise(Frechet; mean = 8.0, sd = 3.0)

    @test native(d) isa Frechet
    @test mean(d)≈8.0 rtol=1e-8
    @test std(d)≈3.0 rtol=1e-8
end
