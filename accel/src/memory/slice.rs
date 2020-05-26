use super::*;
use async_trait::async_trait;
use std::{future::Future, pin::Pin};

/// Typed wrapper of cuPointerGetAttribute
fn get_attr<T, Attr>(ptr: *const T, attr: CUpointer_attribute) -> error::Result<Attr> {
    let mut data = MaybeUninit::<Attr>::uninit();
    unsafe {
        ffi_call!(
            cuPointerGetAttribute,
            data.as_mut_ptr() as *mut c_void,
            attr,
            ptr as CUdeviceptr
        )?;
        Ok(data.assume_init())
    }
}

/// Determine actual memory type dynamically
///
/// Because `Continuous` memories can be treated as a slice,
/// input slice may represents any type of memory.
fn memory_type<T>(ptr: *const T) -> MemoryType {
    match get_attr(ptr, CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE) {
        Ok(CUmemorytype_enum::CU_MEMORYTYPE_HOST) => MemoryType::PageLocked,
        Ok(CUmemorytype_enum::CU_MEMORYTYPE_DEVICE) => MemoryType::Device,
        Ok(CUmemorytype_enum::CU_MEMORYTYPE_ARRAY) => MemoryType::Array,
        Ok(CUmemorytype_enum::CU_MEMORYTYPE_UNIFIED) => {
            unreachable!("CU_POINTER_ATTRIBUTE_MEMORY_TYPE never be UNIFED")
        }
        Err(_) => {
            // unmanaged by CUDA memory system, i.e. host memory
            MemoryType::Host
        }
    }
}

fn get_context<T>(ptr: *const T) -> Option<ContextRef> {
    let ptr =
        get_attr::<_, CUcontext>(ptr, CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT).ok()?;
    Some(ContextRef::from_ptr(ptr))
}

impl<T: Scalar> Memory for [T] {
    type Elem = T;
    fn head_addr(&self) -> *const T {
        self.as_ptr()
    }

    fn head_addr_mut(&mut self) -> *mut T {
        self.as_mut_ptr()
    }

    fn num_elem(&self) -> usize {
        self.len()
    }

    fn memory_type(&self) -> MemoryType {
        memory_type(self.as_ptr())
    }
}

#[async_trait]
impl<T: Scalar> Memcpy<[T]> for [T] {
    fn copy_from(&mut self, src: &[T]) {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.num_elem(), src.num_elem());
        if let Some(ctx) = get_context(self.head_addr()).or_else(|| get_context(src.head_addr())) {
            unsafe {
                contexted_call!(
                    &ctx,
                    cuMemcpy,
                    self.head_addr_mut() as CUdeviceptr,
                    src.as_ptr() as CUdeviceptr,
                    self.num_elem() * T::size_of()
                )
            }
            .unwrap()
        } else {
            self.copy_from_slice(src);
        }
    }

    async fn copy_from_async(&mut self, src: &[T]) {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.num_elem(), src.num_elem());
        let ctx1 = get_context(self.head_addr());
        let ctx2 = get_context(src.head_addr());
        if let Some(ctx) = ctx1.or(ctx2) {
            let stream = stream::Stream::new(ctx);
            let byte_count = self.len() * std::mem::size_of::<T>();
            unsafe {
                contexted_call!(
                    &ctx,
                    cuMemcpyAsync,
                    src.as_ptr() as CUdeviceptr,
                    self.as_mut_ptr() as CUdeviceptr,
                    byte_count,
                    stream.stream
                )
            }
            .expect("Failed to start async memcpy");
            tokio::task::spawn_blocking(move || {
                stream.sync().unwrap();
            })
            .await
            .expect("Async memcpy thread failed");
        } else {
            self.copy_from_slice(src);
        }
    }
}

macro_rules! impl_memcpy_slice {
    ($t:path) => {
        impl<T: Scalar> Memcpy<[T]> for $t {
            fn copy_from(&mut self, src: &[T]) {
                self.as_mut_slice().copy_from(src);
            }
            fn copy_from_async<'a: 'c, 'b: 'c, 'c>(
                &'a mut self,
                src: &'b [T],
            ) -> Pin<Box<dyn Future<Output = ()> + Send + 'c>> {
                self.as_mut_slice().copy_from_async(src)
            }
        }

        impl<T: Scalar> Memcpy<$t> for [T] {
            fn copy_from(&mut self, src: &$t) {
                self.copy_from(src.as_slice());
            }
            fn copy_from_async<'a: 'c, 'b: 'c, 'c>(
                &'a mut self,
                src: &'b $t,
            ) -> Pin<Box<dyn Future<Output = ()> + Send + 'c>> {
                self.copy_from_async(src.as_slice())
            }
        }
    };
}

impl_memcpy_slice!(DeviceMemory::<T>);
impl_memcpy_slice!(PageLockedMemory::<T>);
impl_memcpy_slice!(RegisteredMemory::<'_, T>);

macro_rules! impl_memcpy {
    ($from:path, $to:path) => {
        impl<T: Scalar> Memcpy<$from> for $to {
            fn copy_from(&mut self, src: &$from) {
                self.as_mut_slice().copy_from(src.as_slice());
            }
            fn copy_from_async<'a: 'c, 'b: 'c, 'c>(
                &'a mut self,
                src: &'b $from,
            ) -> Pin<Box<dyn Future<Output = ()> + Send + 'c>> {
                self.as_mut_slice().copy_from_async(src.as_slice())
            }
        }
    };
}

impl_memcpy!(DeviceMemory::<T>, DeviceMemory::<T>);
impl_memcpy!(DeviceMemory::<T>, RegisteredMemory::<'_, T>);
impl_memcpy!(DeviceMemory::<T>, PageLockedMemory::<T>);
impl_memcpy!(PageLockedMemory::<T>, DeviceMemory::<T>);
impl_memcpy!(PageLockedMemory::<T>, RegisteredMemory::<'_, T>);
impl_memcpy!(PageLockedMemory::<T>, PageLockedMemory::<T>);
impl_memcpy!(RegisteredMemory::<'_, T>, DeviceMemory::<T>);
impl_memcpy!(RegisteredMemory::<'_, T>, RegisteredMemory::<'_, T>);
impl_memcpy!(RegisteredMemory::<'_, T>, PageLockedMemory::<T>);

impl<T: Scalar> Continuous for [T] {
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }

    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_type_host_vec() -> error::Result<()> {
        let a = vec![0_u32; 12];
        assert_eq!(a.as_slice().memory_type(), MemoryType::Host);
        assert_eq!(a.as_slice().num_elem(), 12);
        Ok(())
    }

    #[test]
    fn memory_type_host_vec_with_context() -> error::Result<()> {
        let device = Device::nth(0)?;
        let _ctx = device.create_context();
        let a = vec![0_u32; 12];
        assert_eq!(a.as_slice().memory_type(), MemoryType::Host);
        assert_eq!(a.as_slice().num_elem(), 12);
        Ok(())
    }

    #[test]
    fn restore_context() -> error::Result<()> {
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let a = PageLockedMemory::<i32>::zeros(&ctx, 12);
        let ctx_ptr = get_context(a.head_addr()).unwrap();
        assert_eq!(*ctx, ctx_ptr);
        Ok(())
    }

    #[tokio::test]
    async fn memcpy_async_host() {
        let a = vec![1_u32; 12];
        let mut b1 = vec![0_u32; 12];
        let mut b2 = vec![0_u32; 12];
        let mut b3 = vec![0_u32; 12];
        let fut1 = b1.copy_from_async(a.as_slice());
        let fut2 = b2.copy_from_async(a.as_slice());
        let fut3 = b3.copy_from_async(a.as_slice());
        fut3.await;
        fut2.await;
        fut1.await;
        assert_eq!(a, b1);
        assert_eq!(a, b2);
        assert_eq!(a, b3);
    }

    #[tokio::test]
    async fn memcpy_async_d2h() {
        let device = Device::nth(0).unwrap();
        let ctx = device.create_context();
        let a = DeviceMemory::from_elem(&ctx, 12, 1_u32);
        let mut b1 = vec![0_u32; 12];
        let mut b2 = vec![0_u32; 12];
        let mut b3 = vec![0_u32; 12];
        let fut1 = b1.copy_from_async(&a);
        let fut2 = b2.copy_from_async(&a);
        let fut3 = b3.copy_from_async(&a);
        fut3.await;
        fut2.await;
        fut1.await;
        assert_eq!(a.as_slice(), b1.as_slice());
        assert_eq!(a.as_slice(), b2.as_slice());
        assert_eq!(a.as_slice(), b3.as_slice());
    }

    #[tokio::test]
    async fn memcpy_async_h2d() {
        let device = Device::nth(0).unwrap();
        let ctx = device.create_context();
        let a = PageLockedMemory::from_elem(&ctx, 12, 1_u32);
        let mut b1 = DeviceMemory::from_elem(&ctx, 12, 0_u32);
        let mut b2 = DeviceMemory::from_elem(&ctx, 12, 0_u32);
        let mut b3 = DeviceMemory::from_elem(&ctx, 12, 0_u32);
        let fut1 = b1.copy_from_async(&a);
        let fut2 = b2.copy_from_async(&a);
        let fut3 = b3.copy_from_async(&a);
        fut3.await;
        fut2.await;
        fut1.await;
        assert_eq!(a.as_slice(), b1.as_slice());
        assert_eq!(a.as_slice(), b2.as_slice());
        assert_eq!(a.as_slice(), b3.as_slice());
    }

    #[tokio::test]
    async fn memcpy_async_d2d() {
        let device = Device::nth(0).unwrap();
        let ctx = device.create_context();
        let a = DeviceMemory::from_elem(&ctx, 12, 1_u32);
        let mut b1 = DeviceMemory::from_elem(&ctx, 12, 0_u32);
        let mut b2 = DeviceMemory::from_elem(&ctx, 12, 0_u32);
        let mut b3 = DeviceMemory::from_elem(&ctx, 12, 0_u32);
        let fut1 = b1.copy_from_async(&a);
        let fut2 = b2.copy_from_async(&a);
        let fut3 = b3.copy_from_async(&a);
        fut3.await;
        fut2.await;
        fut1.await;
        assert_eq!(a.as_slice(), b1.as_slice());
        assert_eq!(a.as_slice(), b2.as_slice());
        assert_eq!(a.as_slice(), b3.as_slice());
    }
}
