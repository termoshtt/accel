use crate::{error::*, *};
use cuda::*;
use std::{ffi::*, path::*};

/// Represent the resource of CUDA middle-IR (PTX/cubin)
#[derive(Debug)]
pub enum Instruction {
    PTX(CString),
    PTXFile(PathBuf),
    Cubin(Vec<u8>),
    CubinFile(PathBuf),
}

impl Instruction {
    /// Constructor for `Instruction::PTX`
    pub fn ptx(s: &str) -> Instruction {
        let ptx = CString::new(s).expect("Invalid PTX string");
        Instruction::PTX(ptx)
    }

    /// Constructor for `Instruction::Cubin`
    pub fn cubin(sl: &[u8]) -> Instruction {
        Instruction::Cubin(sl.to_vec())
    }

    /// Constructor for `Instruction::PTXFile`
    pub fn ptx_file(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(AccelError::FileNotFound {
                path: path.to_owned(),
            });
        }
        Ok(Instruction::PTXFile(path.to_owned()))
    }

    /// Constructor for `Instruction::CubinFile`
    pub fn cubin_file(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(AccelError::FileNotFound {
                path: path.to_owned(),
            });
        }
        Ok(Instruction::CubinFile(path.to_owned()))
    }
}

impl Instruction {
    /// Get type of PTX/cubin
    pub fn input_type(&self) -> CUjitInputType {
        match *self {
            Instruction::PTX(_) | Instruction::PTXFile(_) => CUjitInputType_enum::CU_JIT_INPUT_PTX,
            Instruction::Cubin(_) | Instruction::CubinFile(_) => {
                CUjitInputType_enum::CU_JIT_INPUT_CUBIN
            }
        }
    }
}
