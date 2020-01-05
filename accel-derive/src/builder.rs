use crate::parser::*;
use failure::*;
use quote::quote;
use std::{
    collections::hash_map::DefaultHasher,
    env, fs,
    hash::*,
    io::{Read, Write},
    path::*,
    process::Command,
};

const NIGHTLY_VERSION: &'static str = "nightly-2020-01-02";

trait CheckRun {
    fn check_run(&mut self) -> Fallible<()>;
}

impl CheckRun for Command {
    fn check_run(&mut self) -> Fallible<()> {
        let output = self.output()?;
        if !output.status.success() {
            println!("{}", std::str::from_utf8(&output.stdout)?);
            eprintln!("{}", std::str::from_utf8(&output.stderr)?);
            bail!("External command failed: {:?}", self);
        }
        Ok(())
    }
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
    Command::new("cargo")
        .args(&[&format!("+{}", NIGHTLY_VERSION), "fmt"])
        .current_dir(&dir)
        .check_run()?;
    Command::new("cargo")
        .args(&[
            &format!("+{}", NIGHTLY_VERSION),
            "build",
            "--release",
            "--target",
            "nvptx64-nvidia-cuda",
        ])
        .current_dir(&dir)
        .check_run()?;

    // Read PTX file
    let mut ptx = fs::File::open(dir.join(format!(
        "target/nvptx64-nvidia-cuda/release/{}.ptx",
        meta.name()
    )))?;
    let mut buf = String::new();
    ptx.read_to_string(&mut buf)?;
    Ok(buf)
}
