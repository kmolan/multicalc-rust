pub mod load;
pub mod problems;
pub mod schema;

// Compiles the workspace README's examples when doctests run, so they cannot drift from the API.
// It lives here rather than in `multicalc` because that crate is published, and its package would
// not carry a file from the repository root.
// Yeah, I know its a hack, but its late and I'm tired so here it stays until I am forced to find a
// more elegant solution.
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
pub struct WorkspaceReadmeExamples;
