use std::path::PathBuf;

mod utils;

#[test]
fn check_cpp_api() {
    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("c-api-tests");
    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let mut source_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    source_dir.extend(["tests", "cpp"]);

    // configure cmake for the tests
    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON");
    let status = cmake_config.status().expect("could not run cmake");
    assert!(status.success(), "failed to run cmake configuration");

    // build the tests with cmake
    let mut cmake_build = utils::cmake_build(&build_dir);
    let status = cmake_build.status().expect("could not run cmake");
    assert!(status.success(), "failed to run cmake build");

    // run the tests
    let mut ctest = utils::ctest(&build_dir);
    let status = ctest.status().expect("could not run ctest");
    assert!(status.success(), "failed to run ctest");
}
