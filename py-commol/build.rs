//! Writes the version and the commit of the build into the extension
//! (`COMMOL_VERSION`, `COMMOL_COMMIT`); `src/build_info.rs` reads them.

#[path = "src/build_info.rs"]
mod build_info;

use std::env;
use std::path::PathBuf;

fn main() {
    let source = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("set by cargo"));
    let cargo_version = env::var("CARGO_PKG_VERSION").expect("set by cargo");
    let git = build_info::GitDir::find(&source);

    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/build_info.rs");
    for file in git.iter().flat_map(build_info::GitDir::watched) {
        println!("cargo::rerun-if-changed={}", file.display());
    }

    let commit = git.and_then(|git| git.commit());
    println!(
        "cargo::rustc-env=COMMOL_VERSION={}",
        build_info::version(&cargo_version, commit.as_deref())
    );
    println!(
        "cargo::rustc-env=COMMOL_COMMIT={}",
        commit.unwrap_or_default()
    );
}
