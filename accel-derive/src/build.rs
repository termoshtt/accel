use glob::glob;
use std::io::{Read, Write};
use std::path::*;
use std::{fs, io, process};
use tempdir::TempDir;

pub use config::{Crate, Depends};

#[derive(Debug, From)]
pub enum CompileError {
    ErrorCode((&'static str, i32)),
    Err(io::Error),
}
pub type Result<T> = ::std::result::Result<T, CompileError>;

/// Compile Rust string into PTX string
pub struct Builder {
    path: PathBuf,
    depends: Depends,
}

impl Builder {
    pub fn new(depends: Depends) -> Self {
        let path = TempDir::new("ptx-builder")
            .expect("Failed to create temporal directory")
            .into_path();
        Self::with_path(&path, depends)
    }

    pub fn with_path<P: AsRef<Path>>(path: P, depends: Depends) -> Self {
        let path = path.as_ref();
        fs::create_dir_all(path.join("src")).unwrap();
        Builder {
            path: path.to_owned(),
            depends: depends,
        }
    }

    /// List of dependencies for `extern crate`
    pub fn crates_for_extern(&self) -> Vec<String> {
        self.depends.iter().map(|c| c.name().replace("-", "_")).collect()
    }

    pub fn compile(&mut self, kernel: &str) -> Result<String> {
        self.generate_config()?;
        self.save(kernel, "src/lib.rs")?;
        self.format()?;
        self.clean();
        self.build()?;
        self.link()?;
        self.load_ptx()
    }

    /// save string as a file on the Builder directory
    fn save(&self, contents: &str, filename: &str) -> Result<()> {
        let mut f = fs::File::create(self.path.join(filename))?;
        f.write(contents.as_bytes())?;
        Ok(())
    }

    fn link(&self) -> Result<()> {
        // extract rlibs using ar x
        let pat_rlib = format!("{}/target/**/deps/*.rlib", self.path.display());
        for path in glob(&pat_rlib).unwrap() {
            let path = path.unwrap();
            process::Command::new("ar")
                .args(&["x", path.file_name().unwrap().to_str().unwrap()])
                .current_dir(path.parent().unwrap())
                .check_run("ar failed")?;
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
            .check_run("llvm-link failed")?;
        // compile bytecode to PTX
        process::Command::new("llc")
            .args(&["-mcpu=sm_20", "kernel.bc", "-o", "kernel.ptx"])
            .current_dir(&self.path)
            .check_run("llc failed")?;
        Ok(())
    }

    fn load_ptx(&self) -> Result<String> {
        let mut f = fs::File::open(self.path.join("kernel.ptx"))?;
        let mut res = String::new();
        f.read_to_string(&mut res).unwrap();
        Ok(res)
    }

    fn generate_config(&self) -> Result<()> {
        self.save(&self.depends.to_string(), "Cargo.toml")?;
        self.save(include_str!("nvptx64-nvidia-cuda.json"), "nvptx64-nvidia-cuda.json")?;
        Ok(())
    }

    fn clean(&self) {
        let path = self.path.join("target");
        match fs::remove_dir_all(&path) {
            Ok(_) => {}
            Err(_) => eprintln!("Already clean (dir = {})", path.display()),
        };
    }

    fn format(&self) -> Result<()> {
        process::Command::new("cargo")
            .args(&["fmt"])
            .current_dir(&self.path)
            .check_run("Format failed")
    }

    fn build(&self) -> Result<()> {
        process::Command::new("xargo")
            .args(&["+nightly", "rustc", "--release", "--target", "nvptx64-nvidia-cuda"])
            .current_dir(&self.path)
            .check_run("xargo failed")
    }
}

trait CheckRun {
    fn check_run(&mut self, comment: &'static str) -> Result<()>;
}

impl CheckRun for process::Command {
    fn check_run(&mut self, comment: &'static str) -> Result<()> {
        let st = self.status()?;
        match st.code() {
            Some(c) => {
                if c != 0 {
                    Err(CompileError::ErrorCode((comment, c)).into())
                } else {
                    Ok(())
                }
            }
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile() {
        let src = r#"
        pub unsafe fn add(a: *const f64, b: *const f64, c: *mut f64, n: usize) {
            let i = accel_core::index();
            if (i as usize) < n {
                *c.offset(i) = *a.offset(i) + *b.offset(i);
            }
        }
        "#;
        let depends = Depends::from(&[Crate::with_version("accel-core", "0.2.0-alpha")]);
        let mut builder = Builder::new(depends);
        let ptx = builder.compile(src).unwrap();
        println!("PTX = {:?}", ptx);
    }
}
