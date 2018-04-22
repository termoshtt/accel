use glob::glob;
use std::io::{Read, Write};
use std::path::*;
use std::{env, fs, io, process};
use tempdir::TempDir;

use config::{to_toml, Crate};

/// Compile Rust string into PTX string
pub struct Builder {
    path: PathBuf,
    crates: Vec<Crate>,
}

impl Builder {
    pub fn new(crates: &[Crate]) -> Self {
        let path = TempDir::new("ptx-builder")
            .expect("Failed to create temporal directory")
            .into_path();
        Self::with_path(&path, crates)
    }

    pub fn with_path<P: AsRef<Path>>(path: P, crates: &[Crate]) -> Self {
        let mut path = path.as_ref().to_owned();
        if path.starts_with("~") {
            let home = env::home_dir().expect("Cannot get home dir");
            path = home.join(path.strip_prefix("~").unwrap());
        }
        fs::create_dir_all(path.join("src")).unwrap();
        Builder {
            path: path,
            crates: crates.to_vec(),
        }
    }

    pub fn exists<P: AsRef<Path>>(path: P) -> Self {
        Self::with_path(path, &[])
    }

    pub fn crates(&self) -> &[Crate] {
        &self.crates
    }

    pub fn compile(&mut self, kernel: &str) -> String {
        self.generate_manifest();
        self.copy_triplet();
        self.save(kernel, "src/lib.rs").expect("Failed to create lib.rs");
        self.format();
        self.clean();
        self.build();
        self.link();
        self.load_ptx()
    }

    pub fn build(&self) {
        process::Command::new("xargo")
            .args(&["+nightly", "rustc", "--release", "--target", "nvptx64-nvidia-cuda"])
            .current_dir(&self.path)
            .check_run()
    }

    pub fn link(&self) {
        // extract rlibs using ar x
        let pat_rlib = format!("{}/target/**/deps/*.rlib", self.path.display());
        for path in glob(&pat_rlib).unwrap() {
            let path = path.unwrap();
            process::Command::new("ar")
                .args(&["x", path.file_name().unwrap().to_str().unwrap()])
                .current_dir(path.parent().unwrap())
                .check_run();
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
            .check_run();
        // compile bytecode to PTX
        process::Command::new("llc")
            .args(&["-mcpu=sm_20", "kernel.bc", "-o", "kernel.ptx"])
            .current_dir(&self.path)
            .check_run();
    }

    pub fn load_ptx(&self) -> String {
        let mut f = fs::File::open(self.path.join("kernel.ptx")).expect("Cannot open PTX file");
        let mut res = String::new();
        f.read_to_string(&mut res).expect("Cannot read PTX file");
        res
    }

    pub fn copy_triplet(&self) {
        self.save(include_str!("nvptx64-nvidia-cuda.json"), "nvptx64-nvidia-cuda.json")
            .expect("Cannot create target triplet");
    }

    pub fn generate_manifest(&self) {
        self.save(&to_toml(&self.crates), "Cargo.toml")
            .expect("Cannot create Cargo.toml");
    }

    /// save string as a file on the Builder directory
    fn save(&self, contents: &str, filename: &str) -> io::Result<()> {
        let mut f = fs::File::create(self.path.join(filename))?;
        f.write(contents.as_bytes())?;
        Ok(())
    }

    fn clean(&self) {
        let path = self.path.join("target");
        match fs::remove_dir_all(&path) {
            Ok(_) => {}
            Err(_) => eprintln!("Already clean (dir = {})", path.display()),
        };
    }

    fn format(&self) {
        process::Command::new("cargo")
            .args(&["fmt"])
            .current_dir(&self.path)
            .check_run()
    }
}

trait CheckRun {
    fn check_run(&mut self);
}

impl CheckRun for process::Command {
    fn check_run(&mut self) {
        info!("Execute subprocess: {:?}", self);
        let st = self.status().expect("Command executaion failed");
        match st.code() {
            Some(c) => {
                if c != 0 {
                    panic!("Subprocess exits with error-code({})", c);
                } else {
                    info!("Subprocess exits normally");
                }
            }
            None => warn!("Subprocess terminated by signal"),
        }
    }
}
