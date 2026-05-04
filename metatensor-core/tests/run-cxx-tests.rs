use std::path::PathBuf;

mod utils;

#[test]
fn run_cxx_tests() {
    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("cxx-tests");
    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let mut source_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    source_dir.extend(["tests", "cpp"]);

    // configure cmake for the tests
    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON");
    let output = cmake_config.output().expect("failed to execute cmake");
    utils::check_output(&output, "configuring tests with cmake");

    // build the tests with cmake
    let mut cmake_build = utils::cmake_build(&build_dir);
    let output = cmake_build.output().expect("failed to execute cmake");
    utils::check_output(&output, "building tests with cmake");

    // run the tests
    let mut ctest = utils::ctest(&build_dir);
    let output = ctest.output().expect("failed to execute ctest");
    utils::check_output(&output, "running tests with ctest");
}
