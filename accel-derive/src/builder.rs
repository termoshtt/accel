use crate::parser::*;
use failure::*;
use quote::quote;
use std::{
    collections::hash_map::DefaultHasher,
    env, fs,
    hash::*,
    io::{Read, Write},
    path::*,
    process::{Command, Stdio},
};

const NIGHTLY_VERSION: &'static str = "nightly-2020-01-01";

/// Setup nightly rustc+cargo and nvptx64-nvidia-cuda target
fn rustup() -> Fallible<()> {
    let st = Command::new("rustup")
        .args(&[
            "toolchain",
            "install",
            NIGHTLY_VERSION,
            "--profile",
            "minimal",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if !st.success() {
        bail!("Cannot get nightly toolchain by rustup: {:?}", st);
    }

    let st = Command::new("rustup")
        .args(&[
            "target",
            "add",
            "nvptx64-nvidia-cuda",
            "--toolchain",
            NIGHTLY_VERSION,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if !st.success() {
        bail!("Cannot get nvptx64-nvidia-cuda target: {:?}", st);
    }

    let st = Command::new("rustup")
        .args(&[
            "component",
            "add",
            "rustfmt",
            "--toolchain",
            NIGHTLY_VERSION,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if !st.success() {
        bail!(
            "Failed to install rustfmt for {}: {:?}",
            NIGHTLY_VERSION,
            st
        );
    }
    Ok(())
}

/// Generate Rust code for nvptx64-nvidia-cuda target from tokens
fn ptx_kernel(func: &syn::ItemFn) -> String {
    let vis = &func.vis;
    let ident = &func.sig.ident;
    let unsafety = &func.sig.unsafety;
    let block = &func.block;

    let fn_token = &func.sig.fn_token;
    let inputs = &func.sig.inputs;
    let output = &func.sig.output;

    let kernel = quote! {
        #![feature(abi_ptx)]
        #![no_std]
        #[no_mangle]
        #vis #unsafety extern "ptx-kernel" #fn_token #ident(#inputs) #output #block
        #[panic_handler]
        fn panic(_: &::core::panic::PanicInfo) -> ! { loop {} }
    };
    kernel.to_string()
}

fn calc_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

fn project_id() -> String {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let hash = calc_hash(&manifest_dir);
    let stem = PathBuf::from(manifest_dir)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    format!("{}-{:x}", stem, hash)
}

pub fn compile_tokens(func: &syn::ItemFn) -> Fallible<String> {
    rustup()?;
    let meta = MetaData::from_token(func)?;

    // Create crate
    let dir = dirs::cache_dir()
        .unwrap()
        .join("accel-derive")
        .join(project_id())
        .join(meta.name());
    fs::create_dir_all(dir.join("src"))?;

    // Generate lib.rs and write into a file
    let mut lib_rs = fs::File::create(dir.join("src/lib.rs"))?;
    lib_rs.write(ptx_kernel(func).as_bytes())?;
    lib_rs.sync_data()?;

    // Generate Cargo.toml
    let mut cargo_toml = fs::File::create(dir.join("Cargo.toml"))?;
    cargo_toml.write(toml::to_string(&meta)?.as_bytes())?;
    cargo_toml.sync_data()?;

    // Build
    let st = Command::new("cargo")
        .args(&[&format!("+{}", NIGHTLY_VERSION), "fmt"])
        .current_dir(&dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if !st.success() {
        bail!("cargo-fmt failed: {:?}", st);
    }
    let output = Command::new("cargo")
        .args(&[
            &format!("+{}", NIGHTLY_VERSION),
            "build",
            "--release",
            "--target",
            "nvptx64-nvidia-cuda",
        ])
        .current_dir(&dir)
        .output()?;
    if !output.status.success() {
        println!("{}", std::str::from_utf8(&output.stdout).unwrap());
        eprintln!("{}", std::str::from_utf8(&output.stderr).unwrap());
        eprintln!("You can find entire generated crate at {}", dir.display());
        bail!("cargo-build failed for {}", meta.name());
    }

    // Read PTX file
    let mut ptx = fs::File::open(dir.join(format!(
        "target/nvptx64-nvidia-cuda/release/{}.ptx",
        meta.name()
    )))?;
    let mut buf = String::new();
    ptx.read_to_string(&mut buf)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    #[test]
    fn rustup() {
        super::rustup().unwrap();
    }
}
