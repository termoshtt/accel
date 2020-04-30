use super::*;
use crate::*;
use cuda::*;
use std::{
    ffi::c_void,
    ops::{Deref, DerefMut},
    sync::Arc,
};

pub struct RegisteredMemory<'a, T> {
    ctx: Arc<Context>,
    mem: &'a mut [T],
}

impl<T> Deref for RegisteredMemory<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.mem
    }
}

impl<T> DerefMut for RegisteredMemory<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.mem
    }
}

impl<T> Drop for RegisteredMemory<'_, T> {
    fn drop(&mut self) {
        if let Err(e) = unsafe {
            contexted_call!(
                &self.ctx,
                cuMemHostUnregister,
                self.mem.as_mut_ptr() as *mut c_void
            )
        } {
            log::error!("Failed to unregister memory: {:?}", e);
        }
    }
}

impl<'a, T: Scalar> RegisteredMemory<'a, T> {
    pub fn new(ctx: Arc<Context>, mem: &'a mut [T]) -> Self {
        unsafe {
            contexted_call!(
                &ctx,
                cuMemHostRegister_v2,
                mem.as_mut_ptr() as *mut c_void,
                mem.len() * T::size_of(),
                0
            )
        }
        .expect("Failed to register host memory into CUDA memory system");
        Self { ctx, mem }
    }
}

impl<T: Scalar> Memory for RegisteredMemory<'_, T> {
    type Elem = T;

    fn head_addr(&self) -> *const T {
        self.mem.as_ptr()
    }

    fn head_addr_mut(&mut self) -> *mut T {
        self.mem.as_mut_ptr()
    }

    fn num_elem(&self) -> usize {
        self.mem.len()
    }

    fn memory_type(&self) -> MemoryType {
        MemoryType::Host
    }
}

impl<T> Contexted for RegisteredMemory<'_, T> {
    fn get_context(&self) -> Arc<Context> {
        self.ctx.clone()
    }
}

impl<T: Scalar> Memcpy<Self> for RegisteredMemory<'_, T> {
    fn copy_from(&mut self, src: &Self) {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.num_elem(), src.num_elem());
        self.copy_from_slice(src)
    }
}

impl<T: Scalar> Memcpy<PageLockedMemory<T>> for RegisteredMemory<'_, T> {
    fn copy_from(&mut self, src: &PageLockedMemory<T>) {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.num_elem(), src.num_elem());
        self.copy_from_slice(src)
    }
}

impl<T: Scalar> Memcpy<DeviceMemory<T>> for RegisteredMemory<'_, T> {
    fn copy_from(&mut self, src: &DeviceMemory<T>) {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.num_elem(), src.num_elem());
        unsafe {
            contexted_call!(
                &self.get_context(),
                cuMemcpyDtoH_v2,
                self.as_mut_ptr() as *mut _,
                src.as_ptr() as CUdeviceptr,
                self.num_elem() * T::size_of()
            )
        }
        .expect("memcpy from Device to registered host memory failed")
    }
}

impl<T: Scalar> Memset for RegisteredMemory<'_, T> {
    fn set(&mut self, value: Self::Elem) {
        self.iter_mut().for_each(|v| *v = value);
    }
}

impl<T: Scalar> Continuous for RegisteredMemory<'_, T> {
    fn as_slice(&self) -> &[T] {
        self
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::*;

    #[test]
    fn memory_type() -> Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let mut a = vec![0_i32; 12];
        let mem = RegisteredMemory::<i32>::new(ctx, &mut a);
        let sl = mem.as_slice();
        assert_eq!(sl.memory_type(), MemoryType::Host);
        Ok(())
    }
}
