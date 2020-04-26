//! CUDA Module (i.e. loaded PTX or cubin)

use crate::{contexted_call, contexted_new, device::*, error::*, *};
use cuda::*;
use num_traits::ToPrimitive;
use std::{ffi::*, path::*, ptr::null_mut, sync::Arc};

/// Size of Block (thread block) in [CUDA thread hierarchy]( http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model )
///
/// Every input integer and float convert into `u32` using [ToPrimitive].
/// If the conversion is impossible, e.g. negative or too large integers, the conversion will panics.
///
/// [ToPrimitive]: https://docs.rs/num-traits/0.2.11/num_traits/cast/trait.ToPrimitive.html
///
/// Examples
/// --------
///
/// - Explicit creation
///
/// ```
/// # use accel::*;
/// let block1d = Block::x(64);
/// assert_eq!(block1d.x, 64);
///
/// let block2d = Block::xy(64, 128);
/// assert_eq!(block2d.x, 64);
/// assert_eq!(block2d.y, 128);
///
/// let block3d = Block::xyz(64, 128, 256);
/// assert_eq!(block3d.x, 64);
/// assert_eq!(block3d.y, 128);
/// assert_eq!(block3d.z, 256);
/// ```
///
/// - From single integer (unsigned and signed)
///
/// ```
/// # use accel::*;
/// let block1d: Block = 64_usize.into();
/// assert_eq!(block1d.x, 64);
///
/// let block1d: Block = 64_i32.into();
/// assert_eq!(block1d.x, 64);
/// ```
///
/// - From tuple
///
/// ```
/// # use accel::*;
/// let block1d: Block = (64,).into();
/// assert_eq!(block1d.x, 64);
///
/// let block2d: Block = (64, 128).into();
/// assert_eq!(block2d.x, 64);
/// assert_eq!(block2d.y, 128);
///
/// let block3d: Block = (64, 128, 256).into();
/// assert_eq!(block3d.x, 64);
/// assert_eq!(block3d.y, 128);
/// assert_eq!(block3d.z, 256);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Block {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Block {
    /// 1D Block
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn x<I: ToPrimitive>(x: I) -> Self {
        Block {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: 1,
            z: 1,
        }
    }

    /// 2D Block
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn xy<I1: ToPrimitive, I2: ToPrimitive>(x: I1, y: I2) -> Self {
        Block {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: y.to_u32().expect("Cannot convert to u32"),
            z: 1,
        }
    }

    /// 3D Block
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn xyz<I1: ToPrimitive, I2: ToPrimitive, I3: ToPrimitive>(x: I1, y: I2, z: I3) -> Self {
        Block {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: y.to_u32().expect("Cannot convert to u32"),
            z: z.to_u32().expect("Cannot convert to u32"),
        }
    }
}

impl<I: ToPrimitive> Into<Block> for (I,) {
    fn into(self) -> Block {
        Block::x(self.0)
    }
}

impl<I1: ToPrimitive, I2: ToPrimitive> Into<Block> for (I1, I2) {
    fn into(self) -> Block {
        Block::xy(self.0, self.1)
    }
}

impl<I1: ToPrimitive, I2: ToPrimitive, I3: ToPrimitive> Into<Block> for (I1, I2, I3) {
    fn into(self) -> Block {
        Block::xyz(self.0, self.1, self.2)
    }
}

macro_rules! impl_into_block {
    ($integer:ty) => {
        impl Into<Block> for $integer {
            fn into(self) -> Block {
                Block::x(self)
            }
        }
    };
}

impl_into_block!(u8);
impl_into_block!(u16);
impl_into_block!(u32);
impl_into_block!(u64);
impl_into_block!(u128);
impl_into_block!(usize);
impl_into_block!(i8);
impl_into_block!(i16);
impl_into_block!(i32);
impl_into_block!(i64);
impl_into_block!(i128);
impl_into_block!(isize);

/// Size of Grid (grid of blocks) in [CUDA thread hierarchy]( http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model )
///
/// Every input integer and float convert into `u32` using [ToPrimitive].
/// If the conversion is impossible, e.g. negative or too large integers, the conversion will panics.
///
/// [ToPrimitive]: https://docs.rs/num-traits/0.2.11/num_traits/cast/trait.ToPrimitive.html
///
/// Examples
/// --------
///
/// - Explicit creation
///
/// ```
/// # use accel::*;
/// let grid1d = Grid::x(64);
/// assert_eq!(grid1d.x, 64);
///
/// let grid2d = Grid::xy(64, 128);
/// assert_eq!(grid2d.x, 64);
/// assert_eq!(grid2d.y, 128);
///
/// let grid3d = Grid::xyz(64, 128, 256);
/// assert_eq!(grid3d.x, 64);
/// assert_eq!(grid3d.y, 128);
/// assert_eq!(grid3d.z, 256);
/// ```
///
/// - From single integer (unsigned and signed)
///
/// ```
/// # use accel::*;
/// let grid1d: Grid = 64_usize.into();
/// assert_eq!(grid1d.x, 64);
///
/// let grid1d: Grid = 64_i32.into();
/// assert_eq!(grid1d.x, 64);
/// ```
///
/// - From tuple
///
/// ```
/// # use accel::*;
/// let grid1d: Grid = (64,).into();
/// assert_eq!(grid1d.x, 64);
///
/// let grid2d: Grid = (64, 128).into();
/// assert_eq!(grid2d.x, 64);
/// assert_eq!(grid2d.y, 128);
///
/// let grid3d: Grid = (64, 128, 256).into();
/// assert_eq!(grid3d.x, 64);
/// assert_eq!(grid3d.y, 128);
/// assert_eq!(grid3d.z, 256);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Grid {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Grid {
    /// 1D Grid
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn x<I: ToPrimitive>(x: I) -> Self {
        Grid {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: 1,
            z: 1,
        }
    }

    /// 2D Grid
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn xy<I1: ToPrimitive, I2: ToPrimitive>(x: I1, y: I2) -> Self {
        Grid {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: y.to_u32().expect("Cannot convert to u32"),
            z: 1,
        }
    }

    /// 3D Grid
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn xyz<I1: ToPrimitive, I2: ToPrimitive, I3: ToPrimitive>(x: I1, y: I2, z: I3) -> Self {
        Grid {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: y.to_u32().expect("Cannot convert to u32"),
            z: z.to_u32().expect("Cannot convert to u32"),
        }
    }
}

impl<I: ToPrimitive> Into<Grid> for (I,) {
    fn into(self) -> Grid {
        Grid::x(self.0)
    }
}

impl<I1: ToPrimitive, I2: ToPrimitive> Into<Grid> for (I1, I2) {
    fn into(self) -> Grid {
        Grid::xy(self.0, self.1)
    }
}

impl<I1: ToPrimitive, I2: ToPrimitive, I3: ToPrimitive> Into<Grid> for (I1, I2, I3) {
    fn into(self) -> Grid {
        Grid::xyz(self.0, self.1, self.2)
    }
}

macro_rules! impl_into_grid {
    ($integer:ty) => {
        impl Into<Grid> for $integer {
            fn into(self) -> Grid {
                Grid::x(self)
            }
        }
    };
}

impl_into_grid!(u8);
impl_into_grid!(u16);
impl_into_grid!(u32);
impl_into_grid!(u64);
impl_into_grid!(u128);
impl_into_grid!(usize);
impl_into_grid!(i8);
impl_into_grid!(i16);
impl_into_grid!(i32);
impl_into_grid!(i64);
impl_into_grid!(i128);
impl_into_grid!(isize);

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

/// CUDA Kernel function
#[derive(Debug)]
pub struct Kernel<'module> {
    func: CUfunction,
    module: &'module Module,
}

impl Contexted for Kernel<'_> {
    fn get_context(&self) -> Arc<Context> {
        self.module.get_context()
    }
}

/// Type which can be sent to the device as kernel argument
///
/// ```
/// # use accel::*;
/// # use std::ffi::*;
/// let a: i32 = 10;
/// let p = &a as *const i32;
/// assert_eq!(
///     DeviceSend::as_ptr(&p),
///     &p as *const *const i32 as *const u8
/// );
/// assert!(std::ptr::eq(
///     unsafe { *(DeviceSend::as_ptr(&p) as *mut *const i32) },
///     p
/// ));
/// ```
pub trait DeviceSend: Sized {
    /// Get the address of this value
    fn as_ptr(&self) -> *const u8 {
        self as *const Self as *const u8
    }
}

// Use default impl
impl<T> DeviceSend for *mut T {}
impl<T> DeviceSend for *const T {}
impl DeviceSend for bool {}
impl DeviceSend for i8 {}
impl DeviceSend for i16 {}
impl DeviceSend for i32 {}
impl DeviceSend for i64 {}
impl DeviceSend for isize {}
impl DeviceSend for u8 {}
impl DeviceSend for u16 {}
impl DeviceSend for u32 {}
impl DeviceSend for u64 {}
impl DeviceSend for usize {}
impl DeviceSend for f32 {}
impl DeviceSend for f64 {}

/// Arbitary number of tuple of kernel arguments
///
/// ```
/// # use accel::*;
/// # use std::ffi::*;
/// let a: i32 = 10;
/// let b: f32 = 1.0;
/// assert_eq!(
///   Arguments::kernel_params(&(&a, &b)),
///   vec![&a as *const i32 as *mut _, &b as *const f32 as *mut _, ]
/// );
/// ```
pub trait Arguments<'arg> {
    /// Get a list of kernel parameters to be passed into [cuLaunchKernel]
    ///
    /// [cuLaunchKernel]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15
    fn kernel_params(&self) -> Vec<*mut c_void>;
}

macro_rules! impl_kernel_parameters {
    ($($name:ident),*; $($num:tt),*) => {
        impl<'arg, $($name : DeviceSend),*> Arguments<'arg> for ($( &'arg $name, )*) {
            fn kernel_params(&self) -> Vec<*mut c_void> {
                vec![$( self.$num.as_ptr() as *mut c_void ),*]
            }
        }
    }
}

impl_kernel_parameters!(;);
impl_kernel_parameters!(D0; 0);
impl_kernel_parameters!(D0, D1; 0, 1);
impl_kernel_parameters!(D0, D1, D2; 0, 1, 2);
impl_kernel_parameters!(D0, D1, D2, D3; 0, 1, 2, 3);
impl_kernel_parameters!(D0, D1, D2, D3, D4; 0, 1, 2, 3, 4);
impl_kernel_parameters!(D0, D1, D2, D3, D4, D5; 0, 1, 2, 3, 4, 5);
impl_kernel_parameters!(D0, D1, D2, D3, D4, D5, D6; 0, 1, 2, 3, 4, 5, 6);
impl_kernel_parameters!(D0, D1, D2, D3, D4, D5, D6, D7; 0, 1, 2, 3, 4, 5, 6, 7);
impl_kernel_parameters!(D0, D1, D2, D3, D4, D5, D6, D7, D8; 0, 1, 2, 3, 4, 5, 6, 7, 8);
impl_kernel_parameters!(D0, D1, D2, D3, D4, D5, D6, D7, D8, D9; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
impl_kernel_parameters!(D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
impl_kernel_parameters!(D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);

/// Typed CUDA Kernel launcher
///
/// This will be automatically implemented in [accel_derive::kernel] for autogenerated wrapper
/// module of [Module].
///
/// ```
/// #[accel_derive::kernel]
/// fn f(a: i32) {}
/// ```
///
/// will create a submodule `f`:
///
/// ```
/// mod f {
///     pub const PTX_STR: &str = "PTX string generated by rustc/nvptx64-nvidia-cuda";
///     pub struct Module(::accel::Module);
///     /* impl Module { ... } */
///     /* impl Launchable for Module { ... } */
/// }
/// ```
///
/// Implementation of `Launchable` for `f::Module` is also generated by [accel_derive::kernel]
/// proc-macro.
///
/// [accel_derive::kernel]: https://docs.rs/accel-derive/0.3.0-alpha.1/accel_derive/attr.kernel.html
/// [Module]: struct.Module.html
pub trait Launchable<'arg> {
    /// Arguments for the kernel to be launched.
    /// This must be a tuple of [DeviceSend] types.
    ///
    /// [DeviceSend]: trait.DeviceSend.html
    type Args: Arguments<'arg>;

    fn get_kernel(&self) -> Result<Kernel>;

    /// Launch CUDA Kernel synchronously
    ///
    /// ```
    /// use accel::*;
    ///
    /// #[accel_derive::kernel]
    /// fn f(a: i32) {}
    ///
    /// let device = Device::nth(0)?;
    /// let ctx = device.create_context();
    /// let module = f::Module::new(ctx)?;
    /// let a = 12;
    /// module.launch((1,) /* grid */, (4,) /* block */, &(&a,))?; // wait until kernel execution ends
    /// # Ok::<(), ::accel::error::AccelError>(())
    /// ```
    fn launch<G: Into<Grid>, B: Into<Block>>(
        &self,
        grid: G,
        block: B,
        args: &Self::Args,
    ) -> Result<()> {
        let grid = grid.into();
        let block = block.into();
        let kernel = self.get_kernel()?;
        let mut params = args.kernel_params();
        unsafe {
            contexted_call!(
                &kernel.get_context(),
                cuLaunchKernel,
                kernel.func,
                grid.x,
                grid.y,
                grid.z,
                block.x,
                block.y,
                block.z,
                0,          /* FIXME: no shared memory */
                null_mut(), /* use default stream */
                params.as_mut_ptr(),
                null_mut() /* no extra */
            )?;
        }
        kernel.sync_context()?;
        Ok(())
    }
}

/// OOP-like wrapper of `cuModule*` APIs
#[derive(Debug)]
pub struct Module {
    module: CUmodule,
    context: Arc<Context>,
}

impl Drop for Module {
    fn drop(&mut self) {
        if let Err(e) = unsafe { contexted_call!(&self.get_context(), cuModuleUnload, self.module) }
        {
            log::error!("Failed to unload module: {:?}", e);
        }
    }
}

impl Contexted for Module {
    fn get_context(&self) -> Arc<Context> {
        self.context.clone()
    }
}

impl Module {
    /// integrated loader of Instruction
    pub fn load(context: Arc<Context>, data: &Instruction) -> Result<Self> {
        match *data {
            Instruction::PTX(ref ptx) => {
                let module = unsafe {
                    contexted_new!(&context, cuModuleLoadData, ptx.as_ptr() as *const _)?
                };
                Ok(Module { module, context })
            }
            Instruction::Cubin(ref bin) => {
                let module = unsafe {
                    contexted_new!(&context, cuModuleLoadData, bin.as_ptr() as *const _)?
                };
                Ok(Module { module, context })
            }
            Instruction::PTXFile(ref path) | Instruction::CubinFile(ref path) => {
                let filename = path_to_cstring(path);
                let module = unsafe { contexted_new!(&context, cuModuleLoad, filename.as_ptr())? };
                Ok(Module { module, context })
            }
        }
    }

    pub fn from_str(context: Arc<Context>, ptx: &str) -> Result<Self> {
        let data = Instruction::ptx(ptx);
        Self::load(context, &data)
    }

    /// Wrapper of `cuModuleGetFunction`
    pub fn get_kernel(&self, name: &str) -> Result<Kernel> {
        let name = CString::new(name).expect("Invalid Kernel name");
        let func = unsafe {
            contexted_new!(
                &self.get_context(),
                cuModuleGetFunction,
                self.module,
                name.as_ptr()
            )
        }?;
        Ok(Kernel { func, module: self })
    }
}

fn path_to_cstring(path: &Path) -> CString {
    CString::new(path.to_str().unwrap()).expect("Invalid Path")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_do_nothing() -> Result<()> {
        // generated by do_nothing example in accel-derive
        let ptx = r#"
        .version 3.2
        .target sm_30
        .address_size 64
        .visible .entry do_nothing()
        {
          ret;
        }
        "#;
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let _mod = Module::from_str(ctx, ptx)?;
        Ok(())
    }
}
