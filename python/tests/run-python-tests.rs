use std::path::PathBuf;
use std::process::Command;

#[test]
fn run_python_tests() {
    let tox = which::which("tox").expect("could not find tox");

    let mut root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    root.pop();

    let mut tox = Command::new(tox);
    tox.arg("--");
    if cfg!(debug_assertions) {
        // assume that debug assertions means that we are building the code
        // in debug mode, even if optimizations could be enabled
        tox.env("METATENSOR_BUILD_TYPE", "debug");
    } else {
        tox.env("METATENSOR_BUILD_TYPE", "release");
    }
    tox.current_dir(&root);
    let status = tox.status().expect("failed to run tox");
    assert!(status.success());
}
