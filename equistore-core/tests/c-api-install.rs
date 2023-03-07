use std::path::PathBuf;
use std::process::Command;

mod utils;

#[test]
fn check_c_api_build_install() {
    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");

    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("c-api-install");

    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    let build_type = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };

    // build and install equistore with cmake
    let mut cmake_config = utils::cmake_config(&cargo_manifest_dir, &build_dir, build_type);

    let mut install_dir = build_dir.clone();
    install_dir.push("usr");
    cmake_config.arg(format!("-DCMAKE_INSTALL_PREFIX={}", install_dir.display()));

    let status = cmake_config.status().expect("cmake configuration failed");
    assert!(status.success());

    let mut cmake_build = utils::cmake_build(&build_dir, build_type);
    cmake_build.arg("--target");
    cmake_build.arg("install");

    let status = cmake_build.status().expect("cmake build failed");
    assert!(status.success());

    // try to use the installed equistore from cmake
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("c-api-sample-project");
    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let mut source_dir = PathBuf::from(&cargo_manifest_dir);
    source_dir.extend(["tests", "cmake-project"]);

    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir, build_type);
    cmake_config.arg(format!("-DCMAKE_PREFIX_PATH={}", install_dir.display()));

    let status = cmake_config.status().expect("cmake configuration failed");
    assert!(status.success());

    let mut cmake_build = utils::cmake_build(&build_dir, build_type);
    let status = cmake_build.status().expect("cmake build failed");
    assert!(status.success());

    let ctest = which::which("ctest").expect("could not find ctest");
    let mut ctest = Command::new(ctest);
    ctest.current_dir(&build_dir);
    ctest.arg("--output-on-failure");
    ctest.arg("--C");
    ctest.arg(build_type);
    let status = ctest.status().expect("failed to run tests");
    assert!(status.success());
}
