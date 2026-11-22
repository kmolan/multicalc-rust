//! Rewriting the generated tables in the benchmark docs.
//!
//! Each doc marks the generated part with a pair of HTML comments. Only the text
//! between them is replaced, so the hand-written prose around it survives.

use std::path::Path;

/// Replaces the text between `begin` and `end` in `path` with `body`, leaving
/// everything outside the markers untouched.
pub fn replace_marked_region(path: &Path, begin: &str, end: &str, body: &str) {
    let content =
        std::fs::read_to_string(path).unwrap_or_else(|err| unreachable!("read {path:?}: {err}"));
    let begin_at = content
        .find(begin)
        .unwrap_or_else(|| unreachable!("no {begin:?} marker in {path:?}"));
    let end_at = content
        .find(end)
        .unwrap_or_else(|| unreachable!("no {end:?} marker in {path:?}"));
    let new = format!(
        "{}\n{body}\n{}",
        &content[..begin_at + begin.len()],
        &content[end_at..]
    );
    std::fs::write(path, new).unwrap_or_else(|err| unreachable!("write {path:?}: {err}"));
}
