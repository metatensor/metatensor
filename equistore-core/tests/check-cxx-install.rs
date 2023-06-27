use std::path::PathBuf;

mod utils;

#[test]
fn check_cxx_install() {
    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");

    // ====================================================================== //
    // build and install equistore with cmake
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("cxx-install");

    let deps_dir = build_dir.join("deps");
    let equistore_dep = deps_dir.join("equistore-core");
    std::fs::create_dir_all(&equistore_dep).expect("failed to create equistore dep dir");

    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // configure cmake for equistore
    let mut cmake_config = utils::cmake_config(&cargo_manifest_dir, &equistore_dep);

    let equistore_cmake_prefix = equistore_dep.join("usr");
    cmake_config.arg(format!("-DCMAKE_INSTALL_PREFIX={}", equistore_cmake_prefix.display()));

    let status = cmake_config.status().expect("could not run cmake");
    assert!(status.success(), "failed to run equistore cmake configuration");

    // build and install equistore
    let mut cmake_build = utils::cmake_build(&equistore_dep);
    cmake_build.arg("--target");
    cmake_build.arg("install");

    let status = cmake_build.status().expect("could not run cmake");
    assert!(status.success(), "failed to run equistore cmake build");

    // ====================================================================== //
    // try to use the installed equistore from cmake
    let mut source_dir = PathBuf::from(&cargo_manifest_dir);
    source_dir.extend(["tests", "cmake-project"]);

    // configure cmake for the test cmake project
    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg(format!("-DCMAKE_PREFIX_PATH={}", equistore_cmake_prefix.display()));
    let status = cmake_config.status().expect("could not run cmake");
    assert!(status.success(), "failed to run test project cmake configuration");

    // build the code, linking to equistore
    let mut cmake_build = utils::cmake_build(&build_dir);
    let status = cmake_build.status().expect("could not run cmake");
    assert!(status.success(), "failed to run test project cmake build");

    // run the executables
    let mut ctest = utils::ctest(&build_dir);
    let status = ctest.status().expect("could not run ctest");
    assert!(status.success(), "failed to run test project tests");
}
