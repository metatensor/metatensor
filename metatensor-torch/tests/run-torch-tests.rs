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

    let metatensor_dep = deps_dir.join("metatensor-core");
    std::fs::create_dir_all(&metatensor_dep).expect("failed to create metatensor dep dir");
    let metatensor_cmake_prefix = utils::setup_metatensor(metatensor_dep);

    let torch_dep = deps_dir.join("torch");
    std::fs::create_dir_all(&torch_dep).expect("failed to create torch dep dir");
    let python = utils::create_python_venv(torch_dep);
    let pytorch_cmake_prefix = utils::setup_pytorch(&python);

    // install metatensor-torch in the python virtualenv, it will be used to
    // generate an example module
    utils::setup_metatensor_torch(&python);

    // ====================================================================== //
    // build the metatensor-torch C++ tests and run them
    let source_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // configure cmake for the tests
    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg("-DMETATENSOR_TORCH_TESTS=ON");
    cmake_config.arg(format!(
        "-DCMAKE_PREFIX_PATH={};{}",
        metatensor_cmake_prefix.display(),
        pytorch_cmake_prefix.display()
    ));
    cmake_config.arg(format!("-DPython_EXECUTABLE={}", python.display()));
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
