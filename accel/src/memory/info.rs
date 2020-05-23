use crate::{contexted_call, device::*};
use cuda::*;

/// Total and Free memory size of the device (in bytes)
#[derive(Debug, Clone, Copy, PartialEq)]
struct MemoryInfo {
    free: usize,
    total: usize,
}

impl MemoryInfo {
    fn get(ctx: Context) -> Self {
        let mut free = 0;
        let mut total = 0;
        unsafe {
            contexted_call!(
                &ctx,
                cuMemGetInfo_v2,
                &mut free as *mut usize,
                &mut total as *mut usize
            )
        }
        .expect("Cannot get memory info");
        MemoryInfo { free, total }
    }
}

/// Get total memory size in bytes of the current device
///
/// Panic
/// ------
/// - when given context is not current
pub fn total_memory(ctx: Context) -> usize {
    MemoryInfo::get(ctx).total
}

/// Get free memory size in bytes of the current device
///
/// Panic
/// ------
/// - when given context is not current
pub fn free_memory(ctx: Context) -> usize {
    MemoryInfo::get(ctx).free
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::*;

    #[test]
    fn info() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let mem_info = MemoryInfo::get(ctx);
        dbg!(&mem_info);
        assert!(mem_info.free > 0);
        assert!(mem_info.total > mem_info.free);
        Ok(())
    }
}
