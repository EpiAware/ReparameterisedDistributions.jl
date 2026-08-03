# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Header content for the release notes page. The managed `make.jl` prepends
# this to the project-root NEWS.md when both exist.

const RELEASE_NOTES_HEADER = """
```@meta
EditURL = "https://github.com/EpiAware/ReparameterisedDistributions.jl/blob/main/NEWS.md"
```

# Release notes

Every release of this package is published as a GitHub release, and most
are covered in full by their auto-generated notes there. Releases that
need more context get a written entry in this repository's `NEWS.md`,
reproduced below.

See [GitHub Releases](https://github.com/EpiAware/ReparameterisedDistributions.jl/releases)
for the complete release history, including releases not detailed below.

"""
