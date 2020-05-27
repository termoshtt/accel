use accel::*;

#[kernel]
unsafe fn add(a: *const f32, b: *const f32, c: *mut f32, n: usize) {
    let i = accel_core::index();
    if (i as usize) < n {
        *c.offset(i) = *a.offset(i) + *b.offset(i);
    }
}

fn main() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();

    let _pf = Profiler::start(&ctx);

    // Allocate memories on GPU
    let n = 1024;
    let mut a = DeviceMemory::<f32>::zeros(&ctx, n);
    let mut b = DeviceMemory::<f32>::zeros(&ctx, n);
    let mut c = DeviceMemory::<f32>::zeros(&ctx, n);

    // Accessible from CPU as usual Rust slice (though this will be slow)
    for i in 0..n {
        a[i] = i as f32;
        b[i] = 2.0 * i as f32;
    }

    // Launch kernel synchronously
    add(
        &ctx,
        1, /* grid */
        n, /* block */
        (&a.as_ptr(), &b.as_ptr(), &c.as_mut_ptr(), &n),
    )
    .expect("Kernel call failed");

    Ok(())
}
