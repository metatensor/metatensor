use std::path::PathBuf;

mod utils;

#[test]
fn check_torch_install() {
    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");

    // ====================================================================== //
    // build and install metatensor-torch with cmake
    let mut torch_tests = PathBuf::from(CARGO_TARGET_TMPDIR);
    torch_tests.push("torch-install");

    let deps_dir = torch_tests.join("deps");

    let metatensor_dep = deps_dir.join("metatensor-core");
    std::fs::create_dir_all(&metatensor_dep).expect("failed to create metatensor dep dir");
    let metatensor_cmake_prefix = utils::setup_metatensor(metatensor_dep);

    let torch_dep = deps_dir.join("torch");
    std::fs::create_dir_all(&torch_dep).expect("failed to create torch dep dir");
    let pytorch_cmake_prefix = utils::setup_pytorch(torch_dep);

    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // configure cmake for metatensor-torch
    let metatensor_torch_dep = deps_dir.join("metatensor-torch");
    let install_prefix = metatensor_torch_dep.join("usr");
    std::fs::create_dir_all(&metatensor_torch_dep).expect("failed to create metatensor-torch dep dir");
    let mut cmake_config = utils::cmake_config(&cargo_manifest_dir, &metatensor_torch_dep);
    cmake_config.arg(format!(
        "-DCMAKE_PREFIX_PATH={};{}",
        metatensor_cmake_prefix.display(),
        pytorch_cmake_prefix.display()
    ));

    cmake_config.arg(format!("-DCMAKE_INSTALL_PREFIX={}", install_prefix.display()));

    // The two properties below handle the RPATH for metatensor_torch, setting it
    // in such a way that we can always load libmetatensor.so and libtorch.so from
    // the location they are found at when compiling metatensor-torch. See
    // https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
    // for more information on CMake RPATH handling
    cmake_config.arg("-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON");
    cmake_config.arg("-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON");

    let status = cmake_config.status().expect("could not run cmake");
    assert!(status.success(), "failed to run metatensor_torch cmake configuration");

    // build and install metatensor-torch
    let mut cmake_build = utils::cmake_build(&metatensor_torch_dep);
    cmake_build.arg("--target");
    cmake_build.arg("install");

    let status = cmake_build.status().expect("could not run cmake");
    assert!(status.success(), "failed to run metatensor_torch cmake build");

    // ====================================================================== //
    // // try to use the installed metatensor-torch from cmake
    let build_dir = torch_tests;

    let mut source_dir = PathBuf::from(&cargo_manifest_dir);
    source_dir.extend(["tests", "cmake-project"]);

    // configure cmake for the test cmake project
    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg(format!(
        "-DCMAKE_PREFIX_PATH={};{};{}",
        metatensor_cmake_prefix.display(),
        pytorch_cmake_prefix.display(),
        install_prefix.display(),
    ));

    let status = cmake_config.status().expect("could not run cmake");
    assert!(status.success(), "failed to run test project cmake configuration");

    // build the code, linking to metatensor-torch
    let mut cmake_build = utils::cmake_build(&build_dir);
    let status = cmake_build.status().expect("could not run cmake");
    assert!(status.success(), "failed to run test project cmake build");

    // run the executables
    let mut ctest = utils::ctest(&build_dir);
    let status = ctest.status().expect("could not run ctest");
    assert!(status.success(), "failed to run test project tests");
}
