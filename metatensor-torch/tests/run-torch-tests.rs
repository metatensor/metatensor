use std::path::PathBuf;

mod utils;

#[test]
fn run_torch_tests() {
    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");
    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // ====================================================================== //
    // setup dependencies for the torch tests

    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("torch-tests");
    let deps_dir = build_dir.join("deps");

    let metatensor_dep = deps_dir.join("metatensor-core");
    let metatensor_source_dir = cargo_manifest_dir.join("..").join("metatensor-core");
    let metatensor_cmake_prefix = utils::setup_metatensor_cmake(&metatensor_source_dir, &metatensor_dep);

    let torch_dep = deps_dir.join("torch");
    std::fs::create_dir_all(&torch_dep).expect("failed to create torch dep dir");
    let python_exe = utils::create_python_venv(torch_dep);
    let pytorch_cmake_prefix = utils::setup_torch_pip(&python_exe);

    // ====================================================================== //
    // build the metatensor-torch C++ tests and run them
    let source_dir = cargo_manifest_dir;

    // configure cmake for the tests
    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg("-DMETATENSOR_TORCH_TESTS=ON");
    cmake_config.arg(format!(
        "-DCMAKE_PREFIX_PATH={};{}",
        metatensor_cmake_prefix.display(),
        pytorch_cmake_prefix.display()
    ));

    let output = cmake_config.output().expect("failed to execute cmake");
    utils::check_output(&output, "configuring torch tests with cmake");

    // build the tests with cmake
    let mut cmake_build = utils::cmake_build(&build_dir);
    let output = cmake_build.output().expect("failed to execute cmake");
    utils::check_output(&output, "building torch tests with cmake");

    // run the tests
    let mut ctest = utils::ctest(&build_dir);
    let output = ctest.output().expect("failed to execute ctest");
    utils::check_output(&output, "running torch tests with ctest");
}
