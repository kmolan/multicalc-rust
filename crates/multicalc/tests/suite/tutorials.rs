use std::fs;
use std::path::Path;

// Every tutorial page is compiled as a doctest through an `include_str!` in
// `src/tutorial_examples.rs`. A page that nobody wires in there still renders on GitHub, but its
// examples never run, so this checks that the folder and the include list agree. The other
// direction — an include naming a page that is not there — is already a compile error.
#[test]
fn every_tutorial_page_is_compiled_as_a_doctest() {
    let manifest_directory = Path::new(env!("CARGO_MANIFEST_DIR"));
    let carrier_source =
        fs::read_to_string(manifest_directory.join("src/tutorial_examples.rs")).unwrap();

    let mut pages: Vec<String> = fs::read_dir(manifest_directory.join("tutorials"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".md"))
        .collect();
    pages.sort();

    assert!(!pages.is_empty(), "no tutorial pages found");

    for page in &pages {
        let include = format!("include_str!(\"../tutorials/{page}\")");
        assert!(
            carrier_source.contains(&include),
            "tutorials/{page} is missing `{include}` in src/tutorial_examples.rs, \
             so its examples never run",
        );
    }
}
