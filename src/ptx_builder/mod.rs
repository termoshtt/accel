
use std::path::*;
use std::io::*;
use std::{fs, env, process};
use glob::glob;
use proc_macro::TokenStream;

/// Compile function into PTX using NVPTX backend
pub fn fn2ptx(_fn: TokenStream) -> String {
    install_rustup_nightly();
    let work = work_dir();
    create_dir(&work);
    install_ptx_builder(&work);
    compile(&work);
    load_str(&get_ptx_path(&work))
}

const PTX_BUILDER_TOML: &'static str = include_str!("Cargo.toml");
const PTX_BUILDER_XARGO: &'static str = include_str!("Xargo.toml");
const PTX_BUILDER_TARGET: &'static str = include_str!("nvptx64-nvidia-cuda.json");
const PTX_BUILDER: &'static str = include_str!("lib.rs.j2");

// japaric/core64 cannot be compiled with recent nightly
// https://github.com/japaric/nvptx/issues/12
const NIGHTLY: &'static str = "nightly-2017-09-01";

fn install(work_dir: &Path, contents: &str, filename: &str) {
    let mut f = fs::File::create(work_dir.join(filename)).unwrap();
    f.write(contents.as_bytes()).unwrap();
}

/// Copy contents to build PTX
fn install_ptx_builder(work: &Path) {
    install(work, PTX_BUILDER_TOML, "Cargo.toml");
    install(work, PTX_BUILDER_XARGO, "Xargo.toml");
    install(work, PTX_BUILDER_TARGET, "nvptx64-nvidia-cuda.json");
    // FIXME
    install(work, PTX_BUILDER, "lib.rs");
}

fn install_rustup_nightly() {
    process::Command::new("rustup")
        .args(&["toolchain", "install", NIGHTLY])
        .stdout(process::Stdio::null())
        .status()
        .unwrap();
}

fn compile(work_dir: &Path) {
    // remove old PTX
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

fn work_dir() -> PathBuf {
    let home = env::home_dir().unwrap();
    let work = home.join(".rust2ptx");
    work.into()
}

fn create_dir(work: &Path) {
    if !work.exists() {
        fs::create_dir_all(&work).unwrap();
        fs::create_dir_all(work.join("src")).unwrap();
    }
}
