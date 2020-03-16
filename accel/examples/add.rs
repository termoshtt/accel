use accel::driver::{device::*, memory::*, *};
use accel_derive::kernel;
use anyhow::Result;

#[kernel]
pub unsafe fn add(a: *const f64, b: *const f64, c: *mut f64, n: usize) {
    let i = accel_core::index();
    if (i as usize) < n {
        *c.offset(i) = *a.offset(i) + *b.offset(i);
    }
}

fn main() -> Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context_auto()?;

    let n = 32;
    let mut a = DeviceMemory::<f64>::managed(&ctx, n, AttachFlag::CU_MEM_ATTACH_GLOBAL)?;
    let mut b = DeviceMemory::<f64>::managed(&ctx, n, AttachFlag::CU_MEM_ATTACH_GLOBAL)?;
    let mut c = DeviceMemory::<f64>::managed(&ctx, n, AttachFlag::CU_MEM_ATTACH_GLOBAL)?;

    for i in 0..n {
        a[i] = i as f64;
        b[i] = 2.0 * i as f64;
    }
    println!("a = {:?}", a.as_slice());
    println!("b = {:?}", b.as_slice());

    let grid = Grid::x(1);
    let block = Block::x(n as u32);
    add(
        &ctx,
        grid,
        block,
        &(&a.as_ptr(), &b.as_ptr(), &c.as_mut_ptr(), &n),
    )
    .expect("Kernel call failed");

    println!("c = {:?}", c.as_slice());
    Ok(())
}
