use std::path::PathBuf;
use std::process::Command;

fn main() {
    let mut metatensor_core = PathBuf::from("../metatensor-core");

    // setting DESTDIR when building with make will cause the install to be in a
    // different directory than the expected one ($OUT_DIR/lib)
    std::env::remove_var("DESTDIR");

    let mut cargo_toml = metatensor_core.clone();
    cargo_toml.push("Cargo.toml");

    // when packaging for crates.io, the metatensor-core symlink is not included.
    // instead, we manually package metatensor-core as a .crate files (actually a
    // .tar.gz file), and unpack it here. We then use cmake to build the code as
    // if it was a standard C library (and cmake calls back cargo to build the
    // rust code in metatensor-core)
    if !cargo_toml.is_file() {
        let cmake_exe = which::which("cmake").expect("could not find cmake");

        let all_crate_files = glob::glob("metatensor-core-cxx-*.tar.gz")
            .expect("bad pattern")
            .flatten()
            .collect::<Vec<_>>();

        if all_crate_files.len() != 1 {
            panic!("could not find the metatensor-core-cxx tarball, run script/update-core.sh");
        }
        let mut crate_file = std::env::current_dir().expect("missing cwd");
        crate_file.push(&all_crate_files[0]);

        metatensor_core = PathBuf::from(std::env::var("OUT_DIR").expect("missing OUT_DIR"));

        Command::new(cmake_exe)
            .arg("-E")
            .arg("tar")
            .arg("xf")
            .arg(&crate_file)
            .current_dir(&metatensor_core)
            .status()
            .expect("failed to unpack metatensor-core");

        let crate_dir = crate_file.file_name().expect("file name").to_str().expect("UTF8 error");
        let splitted = crate_dir.split('.').collect::<Vec<_>>();
        metatensor_core.push(splitted[..splitted.len() - 2].join("."));
    }

    let install_dir = cmake::Config::new(&metatensor_core)
        .define("CARGO_EXE", env!("CARGO"))
        .define("RUST_BUILD_TARGET", std::env::var("TARGET").unwrap())
        .define("BUILD_SHARED_LIBS", if cfg!(feature="static") { "OFF" } else { "ON" })
        .define("METATENSOR_INSTALL_BOTH_STATIC_SHARED", "OFF")
        .build();

    let lib_install_dir = install_dir.join("lib");
    assert!(lib_install_dir.is_dir(), "installation of metatensor-core failed");

    println!("cargo:rustc-link-search=native={}", lib_install_dir.display());

    if cfg!(all(target_os = "windows", not(feature="static"))) {
        // on windows, the DLL is installed in <prefix>/bin, while the link
        // library (.dll.lib) is installed in in <prefix>/lib. We need
        // `lib_install_dir` to find the link library at compile time, and
        // `bin_install_dir` to find the DLL when running tests/etc. from cargo.
        let bin_install_dir = install_dir.join("bin");
        assert!(bin_install_dir.is_dir(), "installation of metatensor-core failed");
        println!("cargo:rustc-link-search=native={}", bin_install_dir.display());
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", metatensor_core.display());
}
