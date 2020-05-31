use accel::*;

#[kernel]
pub unsafe fn read_host_memory(a: *const i32) {
    let i = accel_core::index() as isize;
    accel_core::println!("a[{}] = {}", i, unsafe { *(a.offset(i)) });
}

#[test]
fn page_locked() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();

    let mut a = PageLockedMemory::zeros(&ctx, 4);
    a[0] = 0;
    a[1] = 1;
    a[2] = 2;
    a[3] = 3;
    read_host_memory(&ctx, 1, 4, (a.as_ptr(),))?;
    Ok(())
}

#[test]
fn registered() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();

    let mut a = vec![0; 4];
    let mut mem = RegisteredMemory::new(&ctx, &mut a);
    mem[0] = 0;
    mem[1] = 1;
    mem[2] = 2;
    mem[3] = 3;
    read_host_memory(&ctx, 1, 4, (mem.as_ptr(),))?;
    Ok(())
}
