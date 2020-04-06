//! Memory management
//!
//! There are several types of memories to be considered:
//!
//! - (usual) Host memory
//!   - allocated by usual manner, e.g. `vec![0; n]`
//! - registered Host memory
//!   - allocated by usual manner, and registered into CUDA unified memory system
//! - Page-locked Host memory
//!   - OS memory paging feature is disabled for accelarating memory transfer
//! - Device memory
//!   - allocated on device as a single span
//! - Array
//!   - properly aligned memory on device for using Texture and Surface memory
//!

pub mod array;
pub mod device;

pub use array::*;
pub use device::*;
