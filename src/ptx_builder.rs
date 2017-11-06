
extern crate glob;
extern crate clap;

use std::path::*;
use std::io::*;
use std::{fs, env, process};
use glob::glob;
use clap::{App, Arg};

const PTX_BUILDER_TOML: &'static str = r#"
[package]
name = "ptx-builder"
version = "0.1.0"
[profile.dev]
debug = false
"#;

const PTX_BUILDER_XARGO: &'static str = r#"
[dependencies.core]
git = "https://github.com/japaric/core64"
"#;

const PTX_BUILDER_TARGET: &'static str = r#"
{
    "arch": "nvptx64",
    "cpu": "sm_20",
    "data-layout": "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64",
    "linker": "false",
    "linker-flavor": "ld",
    "llvm-target": "nvptx64-nvidia-cuda",
    "max-atomic-width": 0,
    "os": "cuda",
    "panic-strategy": "abort",
    "target-endian": "little",
    "target-c-int-width": "32",
    "target-pointer-width": "64"
}
"#;

const PTX_BUILDER: &'static str = r#"
#![feature(abi_ptx)]
#![no_std]

#[no_mangle]
pub extern "ptx-kernel" fn bar() {}
"#;

const NIGHTLY: &'static str = "nightly-2017-09-01";

fn generate_ptx_builder(work: &Path) {
    let save = |fname: &str, s: &str| {
        let mut f = fs::File::create(work.join(fname)).unwrap();
        f.write(s.as_bytes()).unwrap();
    };
    save("Cargo.toml", PTX_BUILDER_TOML);
    save("Xargo.toml", PTX_BUILDER_XARGO);
    save("nvptx64-nvidia-cuda.json", PTX_BUILDER_TARGET);
    save("src/lib.rs", PTX_BUILDER);
}

fn install_rustup_nightly() {
    process::Command::new("rustup")
        .args(&["toolchain", "install", NIGHTLY])
        .stdout(process::Stdio::null())
        .status()
        .unwrap();
}

fn compile(work_dir: &Path) {
    process::Command::new("rm")
        .args(&["-rf", "target"])
        .current_dir(work_dir)
        .status()
        .unwrap();
    process::Command::new("xargo")
        .args(
            &[
                "rustc",
                "--release",
                "--target",
                "nvptx64-nvidia-cuda",
                "--",
                "--emit=asm",
            ],
        )
        .current_dir(work_dir)
        .env("RUSTUP_TOOLCHAIN", NIGHTLY)
        .status()
        .unwrap();
}

fn get_ptx_path(work_dir: &Path) -> PathBuf {
    let pattern = work_dir.join("target/**/*.s");
    for entry in glob(pattern.to_str().unwrap()).unwrap() {
        match entry {
            Ok(path) => return path,
            Err(_) => unreachable!(""),
        }
    }
    unreachable!("");
}

fn load_str(path: &Path) -> String {
    let f = fs::File::open(path).unwrap();
    let mut buf = BufReader::new(f);
    let mut v = String::new();
    buf.read_to_string(&mut v).unwrap();
    v
}

pub fn rust2ptx() {
    let app = App::new("rust2ptx").version("0.1.0").arg(
        Arg::with_name("output")
            .help("output path")
            .short("o")
            .long("output")
            .takes_value(true),
    );
    let matches = app.get_matches();
    install_rustup_nightly();
    let work = work_dir();
    if !work.exists() {
        fs::create_dir_all(&work).unwrap();
        fs::create_dir_all(work.join("src")).unwrap();
    }
    generate_ptx_builder(&work);
    compile(&work);
    let ptx_path = get_ptx_path(&work);

    if let Some(output) = matches.value_of("output") {
        // Copy PTX to {output}
        fs::copy(ptx_path, output).unwrap();
    } else {
        // Output PTX to stdout
        let ptx = load_str(&ptx_path);
        println!("{}", ptx);
    }
}

fn work_dir() -> PathBuf {
    let home = env::home_dir().unwrap();
    let work = home.join(".rust2ptx");
    work.into()
}
