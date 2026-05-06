use std::path::PathBuf;
use std::sync::Mutex;

mod utils;

lazy_static::lazy_static! {
    // Make sure only one of the tests below run at the time, since they both
    // try to modify the same files
    static ref LOCK: Mutex<()> = Mutex::new(());
}

/// Check that metatensor-torch can be built and installed with cmake, and that
/// the installed version can be used from another cmake project with
/// `find_package`
#[test]
fn check_torch_install() {
    let _guard = match LOCK.lock() {
        Ok(guard) => guard,
        Err(_) => {
            panic!("another test failed, stopping")
        }
    };

    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");
    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // ====================================================================== //
    // build and install metatensor-torch with cmake
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("torch-install");
    build_dir.push("cmake-find-package");
    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");


    let deps_dir = build_dir.join("deps");

    let metatensor_dep = deps_dir.join("metatensor-core");
    let metatensor_source_dir = cargo_manifest_dir.join("..").join("metatensor-core");
    let metatensor_cmake_prefix = utils::setup_metatensor_cmake(&metatensor_source_dir, &metatensor_dep);

    let torch_dep = deps_dir.join("torch");
    std::fs::create_dir_all(&torch_dep).expect("failed to create torch dep dir");
    let python = utils::create_python_venv(torch_dep);
    let pytorch_cmake_prefix = utils::setup_torch_pip(&python);

    // configure cmake for metatensor-torch
    let metatensor_torch_dep = deps_dir.join("metatensor-torch");

    let cmake_options = vec![
        format!(
            "-DCMAKE_PREFIX_PATH={};{}",
            metatensor_cmake_prefix.display(),
            pytorch_cmake_prefix.display()
        ),
        // The two properties below handle the RPATH for metatensor_torch,
        // setting it in such a way that we can always load libmetatensor.so and
        // libtorch.so from the location they are found at when compiling
        // metatensor-torch. See
        // https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
        // for more information on CMake RPATH handling
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON".into(),
        "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON".into(),
    ];

    let install_prefix = utils::setup_metatensor_torch_cmake(
        &cargo_manifest_dir,
        &metatensor_torch_dep,
        cmake_options,
    );

    // ====================================================================== //
    // // try to use the installed metatensor-torch from cmake
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

    utils::run_command(cmake_config, "cmake configuration");

    // build the code, linking to metatensor-torch
    let cmake_build = utils::cmake_build(&build_dir);
    utils::run_command(cmake_build, "cmake build");

    // run the executables
    let ctest = utils::ctest(&build_dir);
    utils::run_command(ctest, "ctest");
}

/// Same as above, but using pre-built metatensor-torch from the Python wheel,
/// instead of building it from source with cmake.
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
    // build and install metatensor and metatensor-torch with pip
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("torch-install");
    build_dir.push("python-wheels");
    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let mut venv_dir = build_dir.clone();
    venv_dir.push("virtualenv");

    let python_exe = utils::create_python_venv(venv_dir);

    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let python_source_dir = cargo_manifest_dir.parent().unwrap().join("python").join("metatensor_core");
    let metatensor_cmake_prefix = utils::setup_metatensor_pip(&python_exe, &python_source_dir);

    let pytorch_cmake_prefix = utils::setup_torch_pip(&python_exe);

    let python_source_dir = cargo_manifest_dir.parent().unwrap().join("python").join("metatensor_torch");
    let metatensor_torch_cmake_prefix = utils::setup_metatensor_torch_pip(&python_exe, &python_source_dir);

    // ====================================================================== //
    // try to use the installed metatensor-torch from cmake
    let mut source_dir = PathBuf::from(&cargo_manifest_dir);
    source_dir.extend(["tests", "cmake-project"]);

    // configure cmake for the test cmake project
    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg(format!(
        "-DCMAKE_PREFIX_PATH={};{};{}",
        metatensor_cmake_prefix.display(),
        pytorch_cmake_prefix.display(),
        metatensor_torch_cmake_prefix.display(),
    ));

    utils::run_command(cmake_config, "cmake configuration");

    // build the code, linking to metatensor-torch
    let cmake_build = utils::cmake_build(&build_dir);
    utils::run_command(cmake_build, "cmake build");

    // run the executables
    let ctest = utils::ctest(&build_dir);
    utils::run_command(ctest, "ctest");
}

/// Same test as above, but building metatensor and metatensor-torch in the same
/// CMake project (i.e. using add_subdirectory instead of find_package)
#[test]
fn check_cmake_subdirectory() {
    let _guard = match LOCK.lock() {
        Ok(guard) => guard,
        Err(_) => {
            panic!("another test failed, stopping")
        }
    };

    const CARGO_TARGET_TMPDIR: &str = env!("CARGO_TARGET_TMPDIR");

    // install torch
    let mut build_dir = PathBuf::from(CARGO_TARGET_TMPDIR);
    build_dir.push("torch-install");
    build_dir.push("cmake-subdirectory");
    std::fs::create_dir_all(&build_dir).expect("failed to create build dir");

    let deps_dir = build_dir.join("deps");

    let torch_dep = deps_dir.join("torch");
    std::fs::create_dir_all(&torch_dep).expect("failed to create torch dep dir");
    let python = utils::create_python_venv(torch_dep);
    let pytorch_cmake_prefix = utils::setup_torch_pip(&python);

    // ====================================================================== //
    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut source_dir = PathBuf::from(&cargo_manifest_dir);
    source_dir.extend(["tests", "cmake-project"]);

    // configure cmake for the test cmake project
    let mut cmake_config = utils::cmake_config(&source_dir, &build_dir);
    cmake_config.arg(format!("-DCMAKE_PREFIX_PATH={}", pytorch_cmake_prefix.display()));
    cmake_config.arg("-DUSE_CMAKE_SUBDIRECTORY=ON");

    utils::run_command(cmake_config, "cmake configuration");

    // build the code, linking to metatensor-torch
    let cmake_build = utils::cmake_build(&build_dir);
    utils::run_command(cmake_build, "cmake build");

    // run the executables
    let ctest = utils::ctest(&build_dir);
    utils::run_command(ctest, "ctest");
}
