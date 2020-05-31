use accel::*;

#[kernel]
unsafe fn set1(a: *mut i32, n: usize) {
    let i = accel_core::index();
    if i < n as isize {
        *a.offset(i) = 1;
    }
}

#[test]
fn slice_to_pointer_host() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();
    let n = 12;
    let mut a = PageLockedMemory::<i32>::zeros(&ctx, n);
    set1(&ctx, 1, n, (&mut a, n))?;
    assert_eq!(a.as_slice(), vec![1_i32; n].as_slice());
    Ok(())
}

#[test]
fn slice_to_pointer_dev() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();
    let n = 12;
    let mut a = DeviceMemory::<i32>::zeros(&ctx, n);
    set1(&ctx, 1, n, (&mut a, n))?;
    assert_eq!(a.as_slice(), vec![1_i32; n].as_slice());
    Ok(())
}

#[test]
fn slice_to_pointer_registered() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();
    let n = 12;
    let mut v = vec![0_i32; n];
    let mut a = RegisteredMemory::new(&ctx, &mut v);
    set1(&ctx, 1, n, (&mut a, n))?;
    assert_eq!(a.as_slice(), vec![1_i32; n].as_slice());
    Ok(())
}
