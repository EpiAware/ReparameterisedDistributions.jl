# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Header content for the release notes page. The managed `make.jl` prepends
# this to the release notes it renders below.
#
# Keep this header free of any reference to the retired changelog file: the
# kit decides whether to keep a package's custom header by searching it for
# that filename, and replaces the header with a generic one, plus a build
# warning, on a match.

const RELEASE_NOTES_HEADER = """
```@meta
EditURL = "https://github.com/EpiAware/ReparameterisedDistributions.jl/releases"
```

# Release notes

Every release of this package is published as a GitHub release.
The most recent are reproduced below, as they were written there.

"""
