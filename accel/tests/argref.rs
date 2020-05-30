use accel::*;

#[kernel]
fn f(a: &i32, b: &mut i32) {
    if accel_core::index() == 0 {
        *b = *a;
    }
}

#[test]
fn mut_ref_dev() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();
    let mut a = DeviceMemory::<i32>::zeros(&ctx, 1);
    let mut b = DeviceMemory::<i32>::zeros(&ctx, 1);
    a[0] = 1;
    f(&ctx, 1, 1, (&a[0], &mut b[0]))?;
    assert_eq!(a, b);
    Ok(())
}

#[test]
fn mut_ref_host() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();
    let mut a = PageLockedMemory::<i32>::zeros(&ctx, 1);
    let mut b = PageLockedMemory::<i32>::zeros(&ctx, 1);
    a[0] = 1;
    f(&ctx, 1, 1, (&a[0], &mut b[0]))?;
    assert_eq!(a, b);
    Ok(())
}
