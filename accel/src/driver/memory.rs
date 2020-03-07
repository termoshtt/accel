use super::context::*;
use crate::{ffi_call_unsafe, ffi_new_unsafe};
use anyhow::{ensure, Result};
use cuda::*;
use std::mem::MaybeUninit;

pub use cuda::CUmemAttach_flags_enum as AttachFlag;
pub use cuda::CUmemorytype_enum as MemoryType;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryInfo {
    pub free: usize,
    pub total: usize,
}

pub fn get_info(ctx: &Context) -> Result<MemoryInfo> {
    ensure!(ctx.is_current()?, "Given context must be current");
    let mut free = 0;
    let mut total = 0;
    ffi_call_unsafe!(
        cuMemGetInfo_v2,
        &mut free as *mut usize,
        &mut total as *mut usize
    )?;
    Ok(MemoryInfo { free, total })
}

/// low-level wrapper
pub struct DeviceMemory {
    ptr: CUdeviceptr,
    size: usize,
}

impl Drop for DeviceMemory {
    fn drop(&mut self) {
        ffi_call_unsafe!(cuMemFree_v2, self.ptr).expect("Failed to free device memory");
    }
}

impl DeviceMemory {
    pub fn new(ctx: &Context, size: usize) -> Result<Self> {
        ensure!(ctx.is_current()?, "Given context must be current");
        let ptr = ffi_new_unsafe!(cuMemAlloc_v2, size)?;
        Ok(DeviceMemory { ptr, size })
    }

    pub fn managed(ctx: &Context, size: usize, flag: AttachFlag) -> Result<Self> {
        ensure!(ctx.is_current()?, "Given context must be current");
        let ptr = ffi_new_unsafe!(cuMemAllocManaged, size, flag as u32)?;
        Ok(DeviceMemory { ptr, size })
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn memory_type(&self) -> Result<MemoryType> {
        let ty = MaybeUninit::uninit();
        ffi_call_unsafe!(
            cuPointerGetAttribute,
            ty.as_ptr() as *mut _,
            CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            self.ptr
        )?;
        let ty = unsafe { ty.assume_init() };
        Ok(ty)
    }
}

#[cfg(test)]
mod tests {
    use super::super::device::*;
    use super::*;

    #[test]
    fn info() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let mem_info = get_info(&ctx)?;
        dbg!(&mem_info);
        assert!(mem_info.free > 0);
        assert!(mem_info.total > mem_info.free);
        Ok(())
    }

    #[test]
    fn new() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let mem = DeviceMemory::new(&ctx, 12)?;
        assert_eq!(mem.len(), 12);
        Ok(())
    }

    #[test]
    fn memory_type() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context_auto()?;
        let mem = DeviceMemory::new(&ctx, 12)?;
        assert_eq!(mem.memory_type()?, MemoryType::CU_MEMORYTYPE_DEVICE);
        let mem = DeviceMemory::managed(&ctx, 12, AttachFlag::CU_MEM_ATTACH_GLOBAL)?;
        assert_eq!(mem.memory_type()?, MemoryType::CU_MEMORYTYPE_DEVICE);
        Ok(())
    }
}
