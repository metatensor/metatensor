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

    utils::run_command(cmake_config, "cmake configuration");

    // build the tests
    let cmake_build = utils::cmake_build(&build_dir);
    utils::run_command(cmake_build, "cmake build");

    // run the tests
    let ctest = utils::ctest(&build_dir);
    utils::run_command(ctest, "ctest");
}
