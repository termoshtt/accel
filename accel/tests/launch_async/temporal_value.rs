use accel::*;

#[kernel]
unsafe fn add(a: *const f32, b: *const f32, c: *mut f32, n: usize) {
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
    let mut a = DeviceMemory::<f32>::zeros(&ctx, n);
    let mut b = DeviceMemory::<f32>::zeros(&ctx, n);
    let mut c = DeviceMemory::<f32>::zeros(&ctx, n);

    for i in 0..n {
        a[i] = i as f32;
        b[i] = 2.0 * i as f32;
    }

    let md = add::Module::new(&ctx)?;
    let future = md.launch_async(1, n, (&a.as_ptr(), &b.as_ptr(), &c.as_mut_ptr(), &n));
    future.await?;

    Ok(())
}
