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
    utils::run_command(cmake_config, "cmake configuration");

    // build the tests
    let cmake_build = utils::cmake_build(&build_dir);
    utils::run_command(cmake_build, "cmake build");

    // run the tests
    let ctest = utils::ctest(&build_dir);
    utils::run_command(ctest, "ctest");
}
