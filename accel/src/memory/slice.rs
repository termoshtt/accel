use super::*;

impl<'a, T> Memory for &'a [T] {
    type Elem = T;
    fn head_addr(&self) -> *const T {
        self.as_ptr()
    }
    fn byte_size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
}

impl<'a, T> Memory for &'a mut [T] {
    type Elem = T;
    fn head_addr(&self) -> *const T {
        self.as_ptr()
    }
    fn byte_size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
}

impl<'a, T> MemoryMut for &'a mut [T] {
    fn head_addr_mut(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}

impl<'a, T> Continuous for &'a [T] {
    fn length(&self) -> usize {
        self.len()
    }
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

impl<'a, T> Continuous for &'a mut [T] {
    fn length(&self) -> usize {
        self.len()
    }
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

impl<'a, T> ContinuousMut for &'a mut [T] {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_for_slice() -> Result<()> {
        let a = vec![0_u32; 12];
        assert!(matches!(a.as_slice().memory_type(), MemoryType::Host));
        assert_eq!(a.as_slice().length(), 12);
        assert_eq!(a.as_slice().byte_size(), 12 * 4 /* u32 */);
        Ok(())
    }
}
