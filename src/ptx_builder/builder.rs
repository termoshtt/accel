
use super::config::Depends;

use std::path::*;
use std::io::*;
use std::{fs, env, process};
use glob::glob;
use tempdir::TempDir;

const NIGHTLY: &'static str = "nightly-2017-11-18";

pub fn compile(kernel: &str, deps: Depends) -> String {
    let mut builder = Builder::new();
    let ptxs = builder.compile(kernel, deps);
    let mut f = fs::File::open(&ptxs[0]).unwrap();
    let mut res = String::new();
    f.read_to_string(&mut res).unwrap();
    res
}

pub struct Builder {
    path: PathBuf,
    deps: Depends,
}

impl Builder {
    pub fn new() -> Self {
        let path = env::var("ACCEL_PTX_BUILDER_DIR")
            .map(|s| PathBuf::from(&s))
            .or(env::var("CARGO_TARGET_DIR").map(|s| {
                Path::new(&s).join("ptx-builder")
            }))
            .or(TempDir::new("ptx-builder").map(|dir| dir.into_path()))
            .unwrap();
        fs::create_dir_all(path.join("src")).unwrap();
        Builder {
            path,
            deps: Depends::new(),
        }
    }

    pub fn compile(&mut self, kernel: &str, deps: Depends) -> Vec<PathBuf> {
        self.add_depends(deps);
        self.generate_config();
        self.save(kernel, "src/lib.rs");
        self.clean();
        self.build();
        self.ptx_paths()
    }

    fn add_depends(&mut self, mut deps: Depends) {
        self.deps.append(&mut deps);
    }

    /// save string as a file on the Builder directory
    fn save(&self, contents: &str, filename: &str) {
        let mut f = fs::File::create(self.path.join(filename)).unwrap();
        f.write(contents.as_bytes()).unwrap();
    }

    fn ptx_paths(&self) -> Vec<PathBuf> {
        let pattern = self.path.join("target/**/*.s");
        let pattern = pattern.to_str().unwrap();
        glob(pattern).unwrap().map(|x| x.unwrap()).collect()
    }

    fn generate_config(&self) {
        self.save(&self.deps.to_string(), "Cargo.toml");
        self.save(include_str!("Xargo.toml"), "Xargo.toml");
        self.save(
            include_str!("nvptx64-nvidia-cuda.json"),
            "nvptx64-nvidia-cuda.json",
        );
    }

    fn clean(&self) {
        process::Command::new("rm")
            .args(&["-rf", "target"])
            .current_dir(&self.path)
            .status()
            .unwrap();
    }

    fn build(&self) {
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
            .current_dir(&self.path)
            .env("RUSTUP_TOOLCHAIN", NIGHTLY)
            .status()
            .unwrap();
    }
}
