use std::path::PathBuf;

mod utils;

#[test]
fn check_cxx_install() {
    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");

    // ====================================================================== //
    // build and install metatensor with cmake
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("cxx-install");

    let deps_dir = build_dir.join("deps");
    let metatensor_dep = deps_dir.join("metatensor-core");
    std::fs::create_dir_all(&metatensor_dep).expect("failed to create metatensor dep dir");

    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // configure cmake for metatensor
    let mut cmake_config = utils::cmake_config(&cargo_manifest_dir, &metatensor_dep);

    let metatensor_cmake_prefix = metatensor_dep.join("usr");
    cmake_config.arg(format!("-DCMAKE_INSTALL_PREFIX={}", metatensor_cmake_prefix.display()));

    let status = cmake_config.status().expect("could not run cmake");
    assert!(status.success(), "failed to run metatensor cmake configuration");

    // build and install metatensor
    let mut cmake_build = utils::cmake_build(&metatensor_dep);
    cmake_build.arg("--target");
    cmake_build.arg("install");

    let status = cmake_build.status().expect("could not run cmake");
    assert!(status.success(), "failed to run metatensor cmake build");

    // ====================================================================== //
    // try to use the installed metatensor from cmake
    let mut source_dir = PathBuf::from(&cargo_manifest_dir);
    source_dir.extend(["tests", "cmake-project"]);

    // configure cmake for the test cmake project
    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg(format!("-DCMAKE_PREFIX_PATH={}", metatensor_cmake_prefix.display()));
    let status = cmake_config.status().expect("could not run cmake");
    assert!(status.success(), "failed to run test project cmake configuration");

    // build the code, linking to metatensor
    let mut cmake_build = utils::cmake_build(&build_dir);
    let status = cmake_build.status().expect("could not run cmake");
    assert!(status.success(), "failed to run test project cmake build");

    // run the executables
    let mut ctest = utils::ctest(&build_dir);
    let status = ctest.status().expect("could not run ctest");
    assert!(status.success(), "failed to run test project tests");
}
