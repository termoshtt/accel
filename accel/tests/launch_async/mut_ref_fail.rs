use accel::*;

#[kernel]
unsafe fn add(a: *const u32, b: *const u32, c: *mut u32, n: usize) {
    let i = accel_core::index();
    if (i as usize) < n {
        *c.offset(i) = *a.offset(i) + *b.offset(i);
    }
}

#[tokio::main]
async fn main() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();
    let n = 16;
    let mut a = DeviceMemory::<u32>::zeros(&ctx, n);
    let mut b = DeviceMemory::<u32>::zeros(&ctx, n);
    let mut c = DeviceMemory::<u32>::zeros(&ctx, n);

    for i in 0..n {
        a[i] = i as u32;
        b[i] = 2 * i as u32;
    }

    let md = add::Module::new(&ctx)?;
    let future = md.launch_async(1, n, (&a, &b, &mut c, n));

    println!("{:?}", c); // cannot be borrow
    future.await?;

    Ok(())
}
