use std::path::PathBuf;

mod utils;

#[test]
fn run_torch_tests() {
    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");

    // ====================================================================== //
    // setup dependencies for the torch tests

    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("torch-tests");
    let deps_dir = build_dir.join("deps");

    let equistore_dep = deps_dir.join("equistore-core");
    std::fs::create_dir_all(&equistore_dep).expect("failed to create equistore dep dir");
    let equistore_cmake_prefix = utils::setup_equistore(equistore_dep);

    let torch_dep = deps_dir.join("torch");
    std::fs::create_dir_all(&torch_dep).expect("failed to create torch dep dir");
    let pytorch_cmake_prefix = utils::setup_pytorch(torch_dep);

    // ====================================================================== //
    // build the equistore-torch C++ tests and run them
    let mut source_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    source_dir.extend(["..", "equistore-torch"]);

    // configure cmake for the tests
    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg("-DEQUISTORE_TORCH_TESTS=ON");
    cmake_config.arg(format!(
        "-DCMAKE_PREFIX_PATH={};{}",
        equistore_cmake_prefix.display(),
        pytorch_cmake_prefix.display()
    ));
    let status = cmake_config.status().expect("could not run cmake");
    assert!(status.success(), "failed to run torch tests cmake configuration");

    // build the tests with cmake
    let mut cmake_build = utils::cmake_build(&build_dir);
    let status = cmake_build.status().expect("could not run cmake");
    assert!(status.success(), "failed to run torch tests cmake build");

    // run the tests
    let mut ctest = utils::ctest(&build_dir);
    let status = ctest.status().expect("could not run ctest");
    assert!(status.success(), "failed to run running torch tests");
}
