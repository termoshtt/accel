use glob::glob;
use std::io::{Read, Write};
use std::path::*;
use std::{fs, process};
use tempdir::TempDir;

use config::Depends;
use parse::KernelAttribute;

pub struct Builder {
    pub path: PathBuf,
    pub depends: Depends,
}

impl Builder {
    /// Initialize builder with path and dependes
    ///
    /// 1. `build_path` or `build_path_home`
    /// 2. use temporal directory, e.g. "/tmp/ptx-builder.XXXXXXX/"
    pub fn new(attrs: KernelAttribute) -> Self {
        let path = attrs.build_path.unwrap_or(
            TempDir::new("ptx-builder")
                .expect("Failed to create temporal directory")
                .into_path(),
        );
        fs::create_dir_all(path.join("src")).unwrap();
        Builder {
            path: path.to_owned(),
            depends: attrs.depends,
        }
    }

    pub fn compile(&mut self, kernel: &str) -> String {
        self.generate_config();
        self.save(kernel, "src/lib.rs");
        self.format();
        self.clean();
        self.build();
        self.link();
        self.load_ptx()
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
            .arg("panic.ll")
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
        self.save(&self.depends.to_string(), "Cargo.toml");
        self.save(
            include_str!("nvptx64-nvidia-cuda.json"),
            "nvptx64-nvidia-cuda.json",
        );
        self.save(include_str!("panic.ll"), "panic.ll");
    }

    fn clean(&self) {
        process::Command::new("rm")
            .args(&["-rf", "target"])
            .current_dir(&self.path)
            .status()
            .unwrap();
    }

    fn format(&self) {
        process::Command::new("cargo")
            .args(&["fmt"])
            .current_dir(&self.path)
            .status()
            .unwrap();
    }

    fn build(&self) {
        process::Command::new("xargo")
            .args(&[
                "+nightly",
                "rustc",
                "--release",
                "--target",
                "nvptx64-nvidia-cuda",
            ])
            .current_dir(&self.path)
            .status()
            .unwrap();
    }
}
