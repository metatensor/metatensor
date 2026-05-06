#![allow(dead_code)]
#![allow(clippy::needless_return)]

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn build_type() -> &'static str {
    // assume that debug assertion means that we are building the code in
    // debug mode, even if that could be not true in some cases
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn append_flags(existing: Option<String>, extra: &str) -> String {
    match existing {
        Some(flags) if !flags.trim().is_empty() => format!("{flags} {extra}"),
        _ => extra.into(),
    }
}

pub fn cmake_config(source_dir: &Path, build_dir: &Path) -> Command {
    let cmake = which::which("cmake").expect("could not find cmake");

    let mut cmake_config = Command::new(cmake);
    cmake_config.current_dir(build_dir);
    cmake_config.arg(source_dir);
    cmake_config.arg("--no-warn-unused-cli");
    cmake_config.arg(format!("-DCMAKE_BUILD_TYPE={}", build_type()));

    // the cargo executable currently running
    let cargo_exe = std::env::var("CARGO").expect("CARGO env var is not set");
    cmake_config.arg(format!("-DCARGO_EXE={}", cargo_exe));

    if std::env::var_os("CARGO_LLVM_COV").is_some() {
        let coverage_compile_flags = "-fprofile-instr-generate -fcoverage-mapping";
        let coverage_link_flags = "-fprofile-instr-generate";

        let c_flags = append_flags(std::env::var("CFLAGS").ok(), coverage_compile_flags);
        let cxx_flags = append_flags(std::env::var("CXXFLAGS").ok(), coverage_compile_flags);
        let exe_linker_flags =
            append_flags(std::env::var("LDFLAGS").ok(), coverage_link_flags);

        cmake_config.arg(format!("-DCMAKE_C_FLAGS={c_flags}"));
        cmake_config.arg(format!("-DCMAKE_CXX_FLAGS={cxx_flags}"));
        cmake_config.arg(format!("-DCMAKE_EXE_LINKER_FLAGS={exe_linker_flags}"));
        cmake_config.arg(format!("-DCMAKE_SHARED_LINKER_FLAGS={exe_linker_flags}"));
    }

    return cmake_config;
}

pub fn cmake_build(build_dir: &Path) -> Command {
    let cmake = which::which("cmake").expect("could not find cmake");

    let mut cmake_build = Command::new(cmake);
    cmake_build.current_dir(build_dir);
    cmake_build.arg("--build");
    cmake_build.arg(".");
    cmake_build.arg("--parallel");
    cmake_build.arg("--config");
    cmake_build.arg(build_type());

    return cmake_build;
}


pub fn ctest(build_dir: &Path) -> Command {
    let ctest = which::which("ctest").expect("could not find ctest");

    let mut ctest = Command::new(ctest);
    ctest.current_dir(build_dir);
    ctest.arg("--output-on-failure");
    ctest.arg("--build-config");
    ctest.arg(build_type());

    return ctest
}

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

/// Create a fresh Python virtualenv using uv if available, else fallback to
/// `python -m venv`, and return the path to the python executable in the venv
pub fn create_python_venv(build_dir: PathBuf) -> PathBuf {
    if let Some(uv_bin) = find_uv() {
        let mut cmd = Command::new(&uv_bin);
        cmd.arg("venv");
        cmd.arg("--clear");
        cmd.arg(&build_dir);

        run_command(cmd, "uv venv creation");
    } else {
        let mut cmd = Command::new(find_python());
        cmd.arg("-m");
        cmd.arg("venv");
        cmd.arg(&build_dir);

        run_command(cmd, "python to create virtualenv with `venv`");

        // update pip in case the system uses a very old one
        let python = python_in_venv(&build_dir);
        let mut cmd = Command::new(&python);
        cmd.arg("-m");
        cmd.arg("pip");
        cmd.arg("install");
        cmd.arg("--upgrade");
        cmd.arg("pip");

        run_command(cmd, "pip upgrade in virtualenv");
    }

    python_in_venv(&build_dir)
}

#[derive(Default)]
pub struct PipInstallOptions {
    pub upgrade: bool,
    pub no_deps: bool,
    pub no_build_isolation: bool,
}

/// Install a package with pip (uses uv if present, else falls back to python)
fn pip_install(
    python: &Path,
    packages: &[&str],
    options: PipInstallOptions,
) {
    if let Some(uv_bin) = find_uv() {
        let mut cmd = Command::new(&uv_bin);
        cmd.arg("pip").arg("install").arg("--python").arg(python);

        // follow the same behavior as pip when there are multiple indexes
        cmd.arg("--index-strategy");
        cmd.arg("unsafe-best-match");

        if options.upgrade {
            cmd.arg("--upgrade");
        }
        if options.no_deps {
            cmd.arg("--no-deps");
        }
        if options.no_build_isolation {
            cmd.arg("--no-build-isolation");
            // uv doesn't support --check-build-dependencies
        }

        for package in packages {
            cmd.arg(package);
        }

        run_command(cmd, "uv pip install");
    } else {
        let mut cmd = Command::new(python);
        cmd.arg("-m").arg("pip").arg("install");
        if options.upgrade {
            cmd.arg("--upgrade");
        }
        if options.no_deps {
            cmd.arg("--no-deps");
        }
        if options.no_build_isolation {
            // If pip, add both supported options
            cmd.arg("--no-build-isolation");
            cmd.arg("--check-build-dependencies");
        }

        for package in packages {
            cmd.arg(package);
        }

        run_command(cmd, "pip install");
    }
}

/// Download PyTorch in a Python virtualenv, and return the
/// CMAKE_PREFIX_PATH for the corresponding libtorch
pub fn setup_torch_pip(python: &Path) -> PathBuf {
    let torch_version = std::env::var("METATENSOR_TESTS_TORCH_VERSION").unwrap_or("2.11".into());
    pip_install(
        python,
        &[&format!("torch=={}.*", torch_version)],
        PipInstallOptions { upgrade: true, no_deps: false, no_build_isolation: false }
    );

    let mut cmd = Command::new(python);
    cmd.arg("-c");
    cmd.arg("import torch; print(torch.utils.cmake_prefix_path)");

    let output = run_command(cmd, "python to get torch cmake prefix");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let prefix = PathBuf::from(stdout.trim());
    if !prefix.exists() {
        panic!("'torch.utils.cmake_prefix' at '{}' does not exist", prefix.display());
    }

    return prefix;
}

/// Install metatensor in a Python virtualenv with pip, and return the
/// CMAKE_PREFIX_PATH for the installed metatensor.
pub fn setup_metatensor_pip(python: &Path, source_dir: &Path) -> PathBuf {
    pip_install(python, &["setuptools>=77", "packaging>=23", "cmake"], PipInstallOptions::default());

    pip_install(
        python,
        &[&source_dir.display().to_string()],
        PipInstallOptions {
            upgrade: true,
            no_deps: false,
            no_build_isolation: true
        }
    );

    let mut cmd = Command::new(python);
    cmd.arg("-c");
    cmd.arg("import metatensor; print(metatensor.utils.cmake_prefix_path)");

    let output = run_command(cmd, "python to get metatensor cmake prefix");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let prefix = PathBuf::from(stdout.trim());
    if !prefix.exists() {
        panic!("'metatensor.utils.cmake_prefix' at '{}' does not exist", prefix.display());
    }

    return prefix;
}

/// Install metatensor-torch in a Python virtualenv with pip, and return the
/// CMAKE_PREFIX_PATH for the installed metatensor.
pub fn setup_metatensor_torch_pip(python: &Path, source_dir: &Path) -> PathBuf {
    pip_install(
        python,
        &[&source_dir.display().to_string()],
        PipInstallOptions {
            upgrade: true,
            no_deps: true,
            no_build_isolation: true
        }
    );

    let mut cmd = Command::new(python);
    cmd.arg("-c");
    cmd.arg("import metatensor.torch; print(metatensor.torch.utils.cmake_prefix_path)");

    let output = run_command(cmd, "python to get metatensor_torch cmake prefix");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let prefix = PathBuf::from(stdout.trim());
    if !prefix.exists() {
        panic!("'metatensor.torch.utils.cmake_prefix' at '{}' does not exist", prefix.display());
    }

    return prefix;
}

/// Build metatensor-core located in `source_dir` inside `build_dir`, and return
/// the installation prefix
pub fn setup_metatensor_cmake(source_dir: &Path, build_dir: &Path) -> PathBuf {
    std::fs::create_dir_all(build_dir).expect("failed to create metatensor build dir");

    // configure cmake for metatensor
    let mut cmake_config = cmake_config(source_dir, build_dir);

    let install_prefix = build_dir.join("usr");
    cmake_config.arg(format!("-DCMAKE_INSTALL_PREFIX={}", install_prefix.display()));

    run_command(cmake_config, "cmake configuration for metatensor");

    // build and install metatensor
    let mut cmake_build = cmake_build(build_dir);
    cmake_build.arg("--target");
    cmake_build.arg("install");

    run_command(cmake_build, "cmake build for metatensor");

    install_prefix
}


/// Build metatensor-torch located in `source_dir` inside `build_dir`, and return
/// the installation prefix.
pub fn setup_metatensor_torch_cmake(source_dir: &Path, build_dir: &Path, cmake_args: Vec<String>) -> PathBuf {
    std::fs::create_dir_all(build_dir).expect("failed to create metatensor build dir");

    // configure cmake for metatensor
    let mut cmake_config = cmake_config(source_dir, build_dir);

    let install_prefix = build_dir.join("usr");
    cmake_config.arg(format!("-DCMAKE_INSTALL_PREFIX={}", install_prefix.display()));

    // Add any additional cmake arguments
    for arg in cmake_args {
        cmake_config.arg(arg);
    }

    run_command(cmake_config, "cmake configuration for metatensor_torch");

    // build and install metatensor
    let mut cmake_build = cmake_build(build_dir);
    cmake_build.arg("--target");
    cmake_build.arg("install");

    run_command(cmake_build, "cmake build for metatensor_torch");

    install_prefix
}


pub fn run_command(mut command: Command, context: &str) -> std::process::Output {
    write!(std::io::stdout().lock(), "\n\n[Running] {:?}\n\n", command).unwrap();

    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn().unwrap_or_else(|_| panic!("failed to spawn {}", context));

    let mut child_stdout = child.stdout.take().expect("missing stdout");
    let mut child_stderr = child.stderr.take().expect("missing stderr");

    let out_handle = std::thread::spawn(move || -> std::io::Result<Vec<u8>> {
        let mut buf = [0u8; 8192];
        let mut captured = Vec::new();
        let mut sink = std::io::stdout().lock();
        loop {
            let n = child_stdout.read(&mut buf)?;
            if n == 0 {
                break;
            }
            sink.write_all(&buf[..n])?;
            sink.flush()?;
            captured.extend_from_slice(&buf[..n]);
        }
        Ok(captured)
    });

    let err_handle = std::thread::spawn(move || -> std::io::Result<Vec<u8>> {
        let mut buf = [0u8; 8192];
        let mut captured = Vec::new();
        let mut sink = std::io::stderr().lock();
        loop {
            let n = child_stderr.read(&mut buf)?;
            if n == 0 {
                break;
            }
            sink.write_all(&buf[..n])?;
            sink.flush()?;
            captured.extend_from_slice(&buf[..n]);
        }
        Ok(captured)
    });

    let status = child.wait().unwrap_or_else(|_| panic!("failed to run {}", context));
    let stdout = String::from_utf8_lossy(&out_handle.join().unwrap().unwrap()).into_owned();
    let stderr = String::from_utf8_lossy(&err_handle.join().unwrap().unwrap()).into_owned();

    if !status.success() {
        panic!(
            "{} failed, status: {}\nstderr:\n\n{}\nstdout:\n\n{}\n",
            context, status, stderr, stdout
        );
    }

    return std::process::Output { status, stdout: stdout.into_bytes(), stderr: stderr.into_bytes() };
}
