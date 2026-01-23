//! utility functions to run metatensor-torch tests from Cargo
//! (used in `run-torch-tests.rs` and `torch-install-check.rs`)
//! Supports uv and falls back to regular python/pip.

#![allow(clippy::needless_return)]
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "../../../metatensor-core/tests/utils/mod.rs"]
mod core_utils;

pub use core_utils::cmake_build;
pub use core_utils::cmake_config;
pub use core_utils::ctest;

/// Find the path to the uv binary, or None if not present
fn find_uv() -> Option<PathBuf> {
    which::which("uv").ok()
}

/// Find the path to the `python`or `python3` binary on the user system
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
    let python = which::which("python3").expect("failed to run `which python3`");
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

/// Helper: get python executable path inside a venv
fn python_in_venv(venv_dir: &Path) -> PathBuf {
    let mut python = venv_dir.to_path_buf();
    if cfg!(target_os = "windows") {
        python.extend(["Scripts", "python.exe"]);
    } else {
        python.extend(["bin", "python"]);
    }
    python
}

/// Create a fresh Python virtualenv using uv if available, else fallback to python -m venv
pub fn create_python_venv(build_dir: PathBuf) -> PathBuf {
    if let Some(uv_bin) = find_uv() {
        let status = Command::new(&uv_bin)
            .arg("venv")
            .arg("--clear")
            .arg(&build_dir)
            .status()
            .expect("failed to run uv venv");
        assert!(status.success(), "failed to create venv using uv");
    } else {
        let status = Command::new(find_python())
            .arg("-m")
            .arg("venv")
            .arg(&build_dir)
            .status()
            .expect("failed to run python -m venv");
        assert!(status.success(), "failed to create venv using python -m venv");

        // update pip in case the system uses a very old one
        let python = python_in_venv(&build_dir);
        let status = Command::new(&python)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("--upgrade")
            .arg("pip")
            .status()
            .expect("failed to upgrade pip");
        assert!(status.success(), "failed to upgrade pip");
    }

    python_in_venv(&build_dir)
}

/// Helper: pip install (uses uv if present, else falls back to python)
fn pip_install(
    python: &Path,
    packages: &[&str],
    upgrade: bool,
    no_deps: bool,
    no_build_isolation: bool,
) {
    if let Some(uv_bin) = find_uv() {
        let mut cmd = Command::new(&uv_bin);
        cmd.arg("pip").arg("install").arg("--python").arg(python);
        if upgrade {
            cmd.arg("--upgrade");
            // xref: https://github.com/metatensor/metatensor/pull/997#discussion_r2473540252
            cmd.arg("--index-strategy");
            cmd.arg("unsafe-best-match");
        }
        if no_deps {
            cmd.arg("--no-deps");
        }
        if no_build_isolation {
            cmd.arg("--no-build-isolation");
        // uv doesn't support --check-build-dependencies
        }
        for p in packages {
            cmd.arg(p);
        }
        let status = cmd.status().expect("failed to run uv pip install");
        assert!(status.success(), "uv pip install failed");
    } else {
        let mut cmd = Command::new(python);
        cmd.arg("-m").arg("pip").arg("install");
        if upgrade {
            cmd.arg("--upgrade");
        }
        if no_deps {
            cmd.arg("--no-deps");
        }
        if no_build_isolation {
            // If pip, add both supported options
            cmd.arg("--no-build-isolation");
            cmd.arg("--check-build-dependencies");
        }
        for p in packages {
            cmd.arg(p);
        }
        let status = cmd.status().expect("failed to run python pip install");
        assert!(status.success(), "python pip install failed");
    }
}

/// Download PyTorch in a Python virtualenv, and return the
/// CMAKE_PREFIX_PATH for the corresponding libtorch
pub fn setup_pytorch(python: &Path) -> PathBuf {
    let torch_version = std::env::var("METATENSOR_TESTS_TORCH_VERSION").unwrap_or("2.10".into());
    pip_install(
        python,
        &[&format!("torch=={}.*", torch_version)],
        false, // upgrade
        false, // no_deps
        false, // no_build_isolation
    );

    let output = Command::new(python)
        .arg("-c")
        .arg("import torch; print(torch.utils.cmake_prefix_path)")
        .output()
        .expect("failed to run python");
    assert!(output.status.success(), "failed to get torch cmake prefix");

    let stdout = String::from_utf8_lossy(&output.stdout);

    let prefix = PathBuf::from(stdout.trim());
    if !prefix.exists() {
        panic!("'torch.utils.cmake_prefix' at '{}' does not exist", prefix.display());
    }

    prefix
}

/// Install metatensor-torch using the given python
pub fn setup_metatensor_torch(python: &Path) {
    let source_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    pip_install(
        python,
        // this needs to be in sync with pyproject.toml
        &["cmake",
          "packaging>=23",
          "setuptools>=77",
          "numpy"],
        true, // upgrade
        false, // no_deps
        false, // no_build_isolation
    );

    let mut metatensor_core_python = source_dir.clone();
    metatensor_core_python.extend(["..", "python", "metatensor_core"]);
    pip_install(
        python,
        &[metatensor_core_python.to_str().unwrap()],
        false, // upgrade
        true,  // no_deps
        true,  // no_build_isolation
    );

    let mut metatensor_torch_python = source_dir.clone();
    metatensor_torch_python.extend(["..", "python", "metatensor_torch"]);
    pip_install(
        python,
        &[metatensor_torch_python.to_str().unwrap()],
        false, // upgrade
        true,  // no_deps
        true,  // no_build_isolation
    );
}

/// Build metatensor-core in `build_dir`, and return the installation prefix
pub fn setup_metatensor(build_dir: PathBuf) -> PathBuf {
    let mut metatensor_source_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    metatensor_source_dir.extend(["..", "metatensor-core"]);

    // configure cmake for metatensor
    let mut cmake_config = cmake_config(&metatensor_source_dir, &build_dir);

    let install_prefix = build_dir.join("usr");
    cmake_config.arg(format!("-DCMAKE_INSTALL_PREFIX={}", install_prefix.display()));

    let status = cmake_config.status().expect("could not run cmake");
    assert!(status.success(), "failed to run metatensor cmake configuration");

    // build and install metatensor
    let mut cmake_build = cmake_build(&build_dir);
    cmake_build.arg("--target");
    cmake_build.arg("install");

    let status = cmake_build.status().expect("could not run cmake");
    assert!(status.success(), "failed to run metatensor cmake build");

    install_prefix
}
