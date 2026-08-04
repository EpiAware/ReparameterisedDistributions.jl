# PACKAGE-OWNED — the scaffolded quality testset does not call these.
#
# The kit's README wording checks are opt-in: `test_readme_sections` runs from
# the managed `quality.jl`, but the wording set (#292) has to be called by the
# package itself. Without this file the Why section can drift back to prose or
# to a bold-label feature inventory and nothing reports it.

@testitem "Quality: README Why bullets" tags=[:quality, :readme] begin
    using EpiAwarePackageTools
    include(joinpath(@__DIR__, "qa_config.jl"))
    test_readme_bullets(QA_CONFIG.readme.path)
end
