
use super::config::Depends;

use std::path::*;
use std::io::{Read, Write};
use std::{fs, env, process};
use glob::glob;
use tempdir::TempDir;

const NIGHTLY: &'static str = "nightly-2017-11-18";

pub fn compile(kernel: &str, deps: Depends) -> String {
    let mut builder = Builder::new();
    builder.compile(kernel, deps)
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

    pub fn compile(&mut self, kernel: &str, deps: Depends) -> String {
        self.add_depends(deps);
        self.generate_config();
        self.save(kernel, "src/lib.rs");
        self.clean();
        self.build();
        self.link();
        self.load_ptx()
    }

    fn add_depends(&mut self, mut deps: Depends) {
        self.deps.append(&mut deps);
    }

    /// save string as a file on the Builder directory
    fn save(&self, contents: &str, filename: &str) {
        let mut f = fs::File::create(self.path.join(filename)).unwrap();
        f.write(contents.as_bytes()).unwrap();
    }

    fn link(&self) {
        // extract rlibs using ar x
        let pat_rlib = format!("{}/target/**/deps/*.rlib", self.path.display());
        for path in glob(&pat_rlib).unwrap() {
            let path = path.unwrap();
            process::Command::new("ar")
                .args(&["x", path.file_name().unwrap().to_str().unwrap()])
                .current_dir(path.parent().unwrap())
                .status()
                .unwrap();
        }
        // link them
        let pat_rsbc = format!("{}/target/**/deps/*.o", self.path.display());
        let bcs: Vec<_> = glob(&pat_rsbc)
            .unwrap()
            .map(|x| x.unwrap().to_str().unwrap().to_owned())
            .collect();
        process::Command::new("llvm-link")
            .args(&bcs)
            .args(&["-o", "kernel.bc"])
            .current_dir(&self.path)
            .status()
            .unwrap();
        // compile bytecode to PTX
        process::Command::new("llc")
            .args(&["-mcpu=sm_20", "kernel.bc", "-o", "kernel.ptx"])
            .current_dir(&self.path)
            .status()
            .unwrap();
    }

    fn load_ptx(&self) -> String {
        let mut f = fs::File::open(self.path.join("kernel.ptx")).unwrap();
        let mut res = String::new();
        f.read_to_string(&mut res).unwrap();
        res
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
            .args(&["rustc", "--release", "--target", "nvptx64-nvidia-cuda"])
            .current_dir(&self.path)
            .env("RUSTUP_TOOLCHAIN", NIGHTLY)
            .status()
            .unwrap();
    }
}
