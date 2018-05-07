use glob::glob;
use std::io::{Read, Write};
use std::path::*;
use std::{env, fs, io, process, fmt};
use tempdir::TempDir;

use config::{to_toml, Crate};

#[derive(Debug, Clone, Copy)]
pub enum Step {
    Ready,
    Format,
    Link,
    Build,
    Load,
}

#[derive(From)]
pub enum CompileError {
    ExternalCommandError((Step, String, i32)),
    ExternalCommandLaunchError((Step, String, io::Error)),
    IOError((Step, io::Error)),
}
impl fmt::Debug for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            CompileError::ExternalCommandError((step, ref command, ref code)) => {
                write!(f, "External command {} failed during {:?} step. Return code: {}",
                    command, step, code)
            }
            CompileError::ExternalCommandLaunchError((step, ref command, ref err)) => {
                match err.kind() {
                    io::ErrorKind::NotFound => {
                        write!(f, "External command {} failed during {:?} step because the program could not be found. Please ensure this program is installed and try again.",
                            command, step)
                    }
                    _ => {
                        write!(f, "External command {} failed during {:?} step because of an unexpected IO Error: {:?}", 
                        command, step, err)
                    }
                }
            }
            CompileError::IOError((step, ref err)) => {
                write!(f, "Unexpected IO Error during {:?} step: {:?}", step, err)
            }
        }
    }
}
pub type Result<T> = ::std::result::Result<T, CompileError>;

trait Logging {
    type T;
    fn log(self, step: Step) -> Result<Self::T>;
}

impl<T> Logging for io::Result<T> {
    type T = T;
    fn log(self, step: Step) -> Result<Self::T> {
        self.map_err(|e| (step, e).into())
    }
}

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

    pub fn compile(&mut self, kernel: &str) -> Result<String> {
        self.generate_manifest()?;
        self.copy_triplet()?;
        self.save(kernel, "src/lib.rs").log(Step::Ready)?;
        self.format()?;
        self.clean();
        self.build()?;
        self.link()?;
        self.load_ptx()
    }

    pub fn build(&self) -> Result<()> {
        process::Command::new("xargo")
            .args(&["+nightly", "rustc", "--release", "--target", "nvptx64-nvidia-cuda"])
            .current_dir(&self.path)
            .check_run(Step::Build)
    }

    pub fn link(&self) -> Result<()> {
        // extract rlibs using ar x
        let pat_rlib = format!("{}/target/**/deps/*.rlib", self.path.display());
        for path in glob(&pat_rlib).unwrap() {
            let path = path.unwrap();
            process::Command::new("ar")
                .args(&["x", path.file_name().unwrap().to_str().unwrap()])
                .current_dir(path.parent().unwrap())
                .check_run(Step::Link)?;
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
            .check_run(Step::Link)?;
        // compile bytecode to PTX
        process::Command::new("llc")
            .args(&["-mcpu=sm_20", "kernel.bc", "-o", "kernel.ptx"])
            .current_dir(&self.path)
            .check_run(Step::Link)?;
        Ok(())
    }

    pub fn load_ptx(&self) -> Result<String> {
        let mut f = fs::File::open(self.path.join("kernel.ptx")).log(Step::Load)?;
        let mut res = String::new();
        f.read_to_string(&mut res).unwrap();
        Ok(res)
    }

    pub fn copy_triplet(&self) -> Result<()> {
        self.save(include_str!("nvptx64-nvidia-cuda.json"), "nvptx64-nvidia-cuda.json")
            .log(Step::Ready)
    }

    pub fn generate_manifest(&self) -> Result<()> {
        self.save(&to_toml(&self.crates), "Cargo.toml").log(Step::Ready)
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

    fn format(&self) -> Result<()> {
        // TODO: This should not block the build, just warn.
        process::Command::new("cargo")
            .args(&["fmt"])
            .current_dir(&self.path)
            .check_run(Step::Format)
    }
}

trait CheckRun {
    fn check_run(&mut self, step: Step) -> Result<()>;
}

impl CheckRun for process::Command {
    fn check_run(&mut self, step: Step) -> Result<()> {
        let st = self.status().map_err(|e| {
            let command_string = format!("{:?}", self);
            CompileError::ExternalCommandLaunchError((step, command_string, e))
        })?;
        match st.code() {
            Some(c) => {
                if c != 0 {
                    let command_string = format!("{:?}", self);
                    Err(CompileError::ExternalCommandError((step, command_string, c)).into())
                } else {
                    Ok(())
                }
            }
            None => Ok(()),
        }
    }
}
