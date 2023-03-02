use std::path::PathBuf;
use std::process::Command;

fn rustc_version_at_least(version: &str) -> bool {
    rustc_version::version().unwrap() >= rustc_version::Version::parse(version).unwrap()
}

fn main() {
    if cfg!(feature = "static") && !rustc_version_at_least("1.63.0") {
        // We actually only need rustc>=1.63 for tests (because of
        // https://github.com/rust-lang/rust/issues/100066), but there is no way
        // to check if we are building for tests in a build script, so we
        // unconditionally disable the feature for older rustc.
        panic!("'static' feature requires rustc>=1.63");
    }

    let mut equistore_core = PathBuf::from("equistore-core");

    let mut cargo_toml = equistore_core.clone();
    cargo_toml.push("Cargo.toml");

    // when packaging for crates.io, the equistore-core symlink is not included.
    // instead, we manually package equistore-core as a .crate files (actually a
    // .tar.gz file), and unpack it here. We then use cmake to build the code as
    // if it was a standard C library (and cmake calls back cargo to build the
    // rust code in equistore-core)
    if !cargo_toml.is_file() {
        let cmake_exe = which::which("cmake").expect("could not find cmake");

        let all_crate_files = glob::glob("equistore-core-*.crate")
            .expect("bad pattern")
            .flatten()
            .collect::<Vec<_>>();

        if all_crate_files.len() != 1 {
            panic!("could not find the equistore-core crate file, run script/update-core.sh");
        }
        let mut crate_file = std::env::current_dir().expect("missing cwd");
        crate_file.push(&all_crate_files[0]);

        equistore_core = PathBuf::from(std::env::var("OUT_DIR").expect("missing OUT_DIR"));

        Command::new(cmake_exe)
            .arg("-E")
            .arg("tar")
            .arg("xf")
            .arg(&crate_file)
            .current_dir(&equistore_core)
            .status()
            .expect("failed to unpack equistore-core");

        let crate_dir = crate_file.file_name().expect("file name").to_str().expect("UTF8 error");
        let splitted = crate_dir.split('.').collect::<Vec<_>>();
        equistore_core.push(splitted[..splitted.len() - 1].join("."));
    }

    let build = cmake::Config::new(equistore_core)
        .define("CARGO_EXE", env!("CARGO"))
        .build();

    println!("cargo:rustc-link-search=native={}/lib", build.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=equistore-core");
}
