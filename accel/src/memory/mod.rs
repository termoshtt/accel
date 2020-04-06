//! Memory management
//!
//! Unified address
//! ---------------
//!
//! - All memories are mapped into a single 64bit memory space
//!
//! Memory Types
//! ------------
//!
//! |name                     | where exists | From Host | From Device | Description                                                               |
//! |:------------------------|:------------:|:---------:|:-----------:|:--------------------------------------------------------------------------|
//! | (usual) Host memory     | Host         | ✓         |  -          | allocated by usual manner, e.g. `vec![0; n]`                              |
//! | registered Host memory  | Host         | ✓         |  ✓          | allocated by usual manner, and registered into CUDA unified memory system |
//! | Page-locked Host memory | Host         | ✓         |  ✓          | OS memory paging feature is disabled for accelarating memory transfer     |
//! | Device memory           | Device       | ✓         |  ✓          | allocated on device as a single span                                      |
//! | Array                   | Device       | ✓         |  ✓          | properly aligned memory on device for using Texture and Surface memory    |
//!

pub mod array;
pub mod device;

pub use array::*;
pub use device::*;
