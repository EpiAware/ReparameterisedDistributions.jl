# PACKAGE-OWNED — the kit's README wording checks are opt-in, so the managed
# `quality.jl` does not call them (EpiAwarePackageTools#379).

@testitem "Quality: README Why bullets" tags = [:quality, :readme] begin
    using EpiAwarePackageTools
    include(joinpath(@__DIR__, "qa_config.jl"))
    test_readme_bullets(QA_CONFIG.readme.path)
end
