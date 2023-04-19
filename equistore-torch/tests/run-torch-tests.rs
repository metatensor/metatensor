use std::path::PathBuf;
use std::process::Command;

#[path = "../../equistore-core/tests/utils/mod.rs"]
mod utils;


/// Find the path to the python/python3 binary on the user system
fn find_python() -> PathBuf {
    if let Ok(python) = which::which("python") {
        let output = Command::new(&python)
            .arg("-c")
            .arg("import sys; print(sys.version_info.major)")
            .output()
            .expect("could not run python");

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);

            if stdout.trim() == "3" {
                // we found Python 3
                return python;
            }
        }
    }

    // try python3
    let python = which::which("python3").expect("could not find python");
    let output = Command::new(&python)
        .arg("-c")
        .arg("import sys; print(sys.version_info.major)")
        .output()
        .expect("could not run python");

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.trim() == "3" {
            // we found Python 3
            return python;
        }
    }

    panic!("could not find Python 3")
}


/// Download PyTorch in a fresh Python virtualenv, and return the
/// CMAKE_PREFIX_PATH for the corresponding libtorch
fn setup_pytorch(build_dir: PathBuf) -> PathBuf {
    // create virtual env
    let status = Command::new(find_python())
        .arg("-m")
        .arg("venv")
        .arg(&build_dir)
        .status()
        .expect("failed to run python");
    assert!(status.success(), "failed to run `python -m venv`");

    // Use the virtual env python to update pip and install torch
    let mut python = build_dir;
    if cfg!(target_os = "windows") {
        python.extend(["Scripts", "python.exe"]);
    } else {
        python.extend(["bin", "python"]);
    }

    let status = Command::new(&python)
        .arg("-m")
        .arg("pip")
        .arg("install")
        .arg("--upgrade")
        .arg("pip")
        .status()
        .expect("failed to run python");
    assert!(status.success(), "failed to run `python -m pip install --upgrade pip`");

    let torch_version = std::env::var("EQUISTORE_TORCH_TEST_VERSION").unwrap_or("2.0.*".into());
    let status = Command::new(&python)
        .arg("-m")
        .arg("pip")
        .arg("install")
        .arg(format!("torch == {}", torch_version))
        .status()
        .expect("failed to run python");
    assert!(status.success(), "failed to run `python -m pip install torch`");

    let output = Command::new(&python)
        .arg("-c")
        .arg("import torch; print(torch.utils.cmake_prefix_path)")
        .output()
        .expect("failed to run python");
    assert!(output.status.success(), "failed to get torch cmake prefix");

    let stdout = String::from_utf8_lossy(&output.stdout);

    let prefix = PathBuf::from(stdout.trim());
    if !prefix.exists() {
        panic!("'torch.utils.cmake_prefix' at '{}' does not exists", prefix.display());
    }

    prefix
}


#[test]
fn check_torch_api() {
    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");

    // ====================================================================== //
    // build and install equistore with cmake
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("torch-tests-dependencies");
    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());


    let mut equistore_source_dir = cargo_manifest_dir.clone();
    equistore_source_dir.extend(["..", "equistore-core"]);

    // configure cmake for equistore
    let mut cmake_config = utils::cmake_config(&equistore_source_dir, &build_dir);

    let mut equistore_install_dir = build_dir.clone();
    equistore_install_dir.push("usr");
    cmake_config.arg(format!("-DCMAKE_INSTALL_PREFIX={}", equistore_install_dir.display()));

    let status = cmake_config.status().expect("could not run cmake");
    assert!(status.success(), "failed to run equistore cmake configuration");

    // build and install equistore
    let mut cmake_build = utils::cmake_build(&build_dir);
    cmake_build.arg("--target");
    cmake_build.arg("install");

    let status = cmake_build.status().expect("could not run cmake");
    assert!(status.success(), "failed to run equistore cmake build");

    let pytorch_cmake_prefix = setup_pytorch(build_dir.clone());

    // ====================================================================== //
    // build the equistore-torch C++ tests and run them
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("torch-tests");
    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let mut source_dir = cargo_manifest_dir;
    source_dir.extend(["..", "equistore-torch"]);

    // configure cmake for the tests
    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON");
    cmake_config.arg("-DEQUISTORE_TORCH_TESTS=ON");
    cmake_config.arg(format!(
        "-DCMAKE_PREFIX_PATH={};{}",
        equistore_install_dir.display(),
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
