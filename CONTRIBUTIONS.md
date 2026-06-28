# Contributions

multicalc follows [Gitflow workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).
A few simple rules:

1. Do not work on `main` branch directly.
2. Use `features/<features_to_improve>` branch. For example, if you want to improve `numerical_integration` then create `features/numerical_integration` on latest `main` branch & create pull request to `main` branch.
3. Test before push. Use `cargo test --features <required_features>`
4. Update the changelog in the same PR. Add an entry under `## [Unreleased]` in [CHANGELOG.md](./CHANGELOG.md), grouped under `Added` / `Changed` / `Fixed` / `Removed`, for any behavior-facing change.

## Releasing

Releases are automated from `main`:

1. In a PR, bump `version` in `Cargo.toml` and rename the `## [Unreleased]` heading in `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD` (matching the new version), then open a fresh empty `## [Unreleased]` section above it. Refresh the comparison/tag links at the bottom of the changelog (point `[Unreleased]` at `vX.Y.Z...HEAD` and add a `[X.Y.Z]` tag link).
2. When that PR merges, the release workflow publishes to crates.io and creates a `vX.Y.Z` tag and GitHub release whose notes come from the matching changelog section.
3. The `Cargo.toml` version and the top dated changelog heading must match; a bump with no changelog entry fails the release.

Thank you for all the contributions!
