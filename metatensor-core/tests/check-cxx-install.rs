use std::path::PathBuf;
use std::sync::Mutex;

mod utils;

lazy_static::lazy_static! {
    // Make sure only one of the tests below run at the time, since they both
    // try to modify the same files
    static ref LOCK: Mutex<()> = Mutex::new(());
}


/// Check that metatensor can be built and installed with cmake, and that the
/// installed version can be used from another cmake project with `find_package`
#[test]
fn check_cxx_install() {
    let _guard = match LOCK.lock() {
        Ok(guard) => guard,
        Err(_) => {
            panic!("another test failed, stopping")
        }
    };

    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");

    // ====================================================================== //
    // build and install metatensor with cmake
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("cxx-install");
    build_dir.push("cmake-find-package");
    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let deps_dir = build_dir.join("deps");
    let metatensor_dep = deps_dir.join("metatensor-core");

    let source_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let metatensor_cmake_prefix = utils::setup_metatensor_cmake(&source_dir, &metatensor_dep);

    // ====================================================================== //
    // try to use the installed metatensor from cmake
    let mut tests_source_dir = source_dir;
    tests_source_dir.extend(["tests", "cmake-project"]);

    // configure cmake for the test cmake project
    let mut cmake_config = utils::cmake_config(&tests_source_dir, &build_dir);
    cmake_config.arg(format!("-DCMAKE_PREFIX_PATH={}", metatensor_cmake_prefix.display()));
    let output = cmake_config.output().expect("failed to execute cmake");
    utils::check_output(&output, "configuring test project with cmake");

    // build the code, linking to metatensor
    let mut cmake_build = utils::cmake_build(&build_dir);
    let output = cmake_build.output().expect("failed to execute cmake");
    utils::check_output(&output, "building test project with cmake");

    // run the executables
    let mut ctest = utils::ctest(&build_dir);
    let output = ctest.output().expect("failed to execute ctest");
    utils::check_output(&output, "running tests with ctest");
}


/// Same test as above, but using pre-built metatensor from the Python wheel
#[test]
fn check_python_install() {
    let _guard = match LOCK.lock() {
        Ok(guard) => guard,
        Err(_) => {
            panic!("another test failed, stopping")
        }
    };

    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");

    // ====================================================================== //
    // build and install metatensor with pip
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("cxx-install");
    build_dir.push("python-wheels");
    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let mut venv_dir = build_dir.clone();
    venv_dir.push("virtualenv");

    let python_exe = utils::create_python_venv(venv_dir);

    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let python_source_dir = cargo_manifest_dir.parent().unwrap().join("python").join("metatensor_core");

    let metatensor_cmake_prefix = utils::setup_metatensor_pip(&python_exe, &python_source_dir);


    // ====================================================================== //
    // try to use the installed metatensor from cmake
    let mut source_dir = PathBuf::from(&cargo_manifest_dir);
    source_dir.extend(["tests", "cmake-project"]);

    // configure cmake for the test cmake project
    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg(format!("-DCMAKE_PREFIX_PATH={}", metatensor_cmake_prefix.display()));
    let output = cmake_config.output().expect("failed to execute cmake");
    utils::check_output(&output, "configuring test project with cmake");

    // build the code, linking to metatensor
    let mut cmake_build = utils::cmake_build(&build_dir);
    let output = cmake_build.output().expect("failed to execute cmake");
    utils::check_output(&output, "building test project with cmake");

    // run the executables
    let mut ctest = utils::ctest(&build_dir);
    let output = ctest.output().expect("failed to execute ctest");
    utils::check_output(&output, "running tests with ctest");
}

/// Same test as above, but building metatensor in the same CMake project as the
/// test executable (i.e. using add_subdirectory instead of find_package)
#[test]
fn check_cmake_subdirectory() {
    let _guard = match LOCK.lock() {
        Ok(guard) => guard,
        Err(_) => {
            panic!("another test failed, stopping")
        }
    };

    // Same test as above, but building metatensor in the same CMake project as
    // the test executable
    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("cxx-install");
    build_dir.push("cmake-subdirectory");
    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut source_dir = PathBuf::from(&cargo_manifest_dir);
    source_dir.extend(["tests", "cmake-project"]);

    // configure cmake for the test cmake project
    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg("-DUSE_CMAKE_SUBDIRECTORY=ON");

    let output = cmake_config.output().expect("failed to execute cmake");
    utils::check_output(&output, "configuring test project with cmake");

    // build the code, linking to metatensor-torch
    let mut cmake_build = utils::cmake_build(&build_dir);
    let output = cmake_build.output().expect("failed to execute cmake");
    utils::check_output(&output, "building test project with cmake");

    // run the executables
    let mut ctest = utils::ctest(&build_dir);
    let output = ctest.output().expect("failed to execute ctest");
    utils::check_output(&output, "running tests with ctest");
}
