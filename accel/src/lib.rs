//! GPGPU framework for Rust based on [CUDA Driver API]
//!
//! [CUDA Driver API]: https://docs.nvidia.com/cuda/cuda-driver-api/
//!
//! Setup
//! -----
//! Currently (0.3.0), accel works only on Linux system. Windows support will come in future release (0.3.x or 0.4~).
//!
//! 1. Install [CUDA](https://developer.nvidia.com/cuda-downloads) on your system
//! 2. Setup Rust environement using rustup (Requires 1.42 or later)
//! 3. Add `nvptx64-nvidia-cuda` target and install `ptx-linker`, or run
//!
//!     ```shell
//!     curl -sSL https://gitlab.com/termoshtt/accel/raw/master/setup_nvptx_toolchain.sh | bash
//!     ```
//!
//! Basic Examples
//! --------------
//! accel works with stable Rust
//!
//! ```toml
//! [dependencies]
//! accel = "=0.3.0-alpha.2"
//! ```
//!
//! Do **NOT** add `accel-core` to `[dependencies]`.
//! It will be linked automatically into the device code.
//!
//! ### Vector Add
//!
//! ```
//! use accel::*;
//!
//! #[kernel]
//! unsafe fn add(a: *const f64, b: *const f64, c: *mut f64, n: usize) {
//!     let i = accel_core::index();
//!     if (i as usize) < n {
//!         *c.offset(i) = *a.offset(i) + *b.offset(i);
//!     }
//! }
//!
//! fn main() -> error::Result<()> {
//!     let device = Device::nth(0)?;
//!     let ctx = device.create_context();
//!
//!     // Allocate memories on GPU
//!     let n = 32;
//!     let mut a = DeviceMemory::<f64>::zeros(ctx.clone(), n);
//!     let mut b = DeviceMemory::<f64>::zeros(ctx.clone(), n);
//!     let mut c = DeviceMemory::<f64>::zeros(ctx.clone(), n);
//!
//!     // Accessible from CPU as usual Rust slice (though this will be slow)
//!     for i in 0..n {
//!         a[i] = i as f64;
//!         b[i] = 2.0 * i as f64;
//!     }
//!     println!("a = {:?}", a.as_slice());
//!     println!("b = {:?}", b.as_slice());
//!
//!     // Launch kernel synchronously
//!     add(ctx,
//!         1 /* grid */,
//!         n /* block */,
//!         &(&a.as_ptr(), &b.as_ptr(), &c.as_mut_ptr(), &n)
//!     ).expect("Kernel call failed");
//!
//!     println!("c = {:?}", c.as_slice());
//!     Ok(())
//! }
//! ```
//!
//! ### Assertion on GPU
//!
//! ```
//! use accel::*;
//!
//! #[kernel]
//! fn assert() {
//!     accel_core::assert_eq!(1 + 2, 4);  // will fail
//! }
//!
//! fn main() -> error::Result<()> {
//!     let device = Device::nth(0)?;
//!     let ctx = device.create_context();
//!     let result = assert(ctx, 1 /* grid */, 4 /* block */, &());
//!     assert!(result.is_err()); // assertion failed
//!     Ok(())
//! }
//! ```
//!
//! ### Print from GPU
//!
//! ```
//! use accel::*;
//!
//! #[kernel]
//! pub fn print() {
//!     let i = accel_core::index();
//!     accel_core::println!("Hello from {}", i);
//! }
//!
//! fn main() -> error::Result<()> {
//!     let device = Device::nth(0)?;
//!     let ctx = device.create_context();
//!     print(ctx, 1, 4, &())?;
//!     Ok(())
//! }
//! ```
//!
//! Advanced Examples
//! -----------------
//!
//! ### Get compiled PTX as `String`
//!
//! The proc-macro `#[kernel]` creates a submodule `add::` in addition to a function `add`.
//! Kernel Rust code is compiled into PTX string using rustc's `nvptx64-nvidia-cuda` toolchain.
//! Generated PTX string is embedded into proc-macro output as `{kernel_name}::PTX_STR`.
//!
//! ```
//! use accel::kernel;
//!
//! #[kernel]
//! unsafe fn add(a: *const f64, b: *const f64, c: *mut f64, n: usize) {
//!     let i = accel_core::index();
//!     if (i as usize) < n {
//!         *c.offset(i) = *a.offset(i) + *b.offset(i);
//!     }
//! }
//!
//! // PTX assembler code is embedded as `add::PTX_STR`
//! println!("{}", add::PTX_STR);
//! ```
//!
//! ### Asynchronous launch
//!
//! `#[kernel]` creates `assert::Module` type definition which implements [Launchable] trait.
//! This struct will read `PTX_STR` using [Module].
//!
//! [Module]:     ./module/struct.Module.html
//! [Launchable]: ./module/trait.Launchable.html
//!
//! ```
//! use accel::*;
//!
//! #[kernel]
//! fn assert() {
//!     accel_core::assert_eq!(1 + 2, 4);
//! }
//!
//! fn main() -> error::Result<()> {
//!     let device = Device::nth(0)?;
//!     let ctx = device.create_context();
//!     let stream = Stream::new(ctx.clone());
//!
//!     let module = assert::Module::new(ctx)?;
//!     module.stream_launch(&stream, 1, 4, &())?; // lanch will succeed
//!     assert!(stream.sync().is_err()); // assertion failed is detected in next sync
//!     Ok(())
//! }
//! ```

extern crate cuda_driver_sys as cuda;

pub use accel_derive::kernel;

pub mod device;
pub mod error;
pub mod linker;
pub mod memory;
pub mod module;
pub mod stream;

pub use device::*;
pub use linker::*;
pub use memory::*;
pub use module::*;
pub use stream::*;
