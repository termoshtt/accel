use accel::*;
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
    let n = 32;
    let mut a = MVec::new(n)?;
    let mut b = MVec::new(n)?;
    let mut c = MVec::new(n)?;

    let mut a_data = Vec::new();
    let mut b_data = Vec::new();
    for i in 0..n {
        a_data.push(i as f64);
        b_data.push(2.0 * i as f64);
    }
    a.set(a_data.as_slice())?;
    b.set(b_data.as_slice())?;
    println!("a = {:?}", a_data);
    println!("b = {:?}", b_data);

    let device = driver::Device::nth(0)?;
    let ctx = device.create_context_auto()?;
    let grid = Grid::x(1);
    let block = Block::x(n as u32);
    add(
        &ctx,
        grid,
        block,
        &(&a.as_ptr(), &b.as_ptr(), &c.as_mut_ptr(), &n),
    )
    .expect("Kernel call failed");

    device::sync()?;

    let mut c_data = vec![0f64; n];
    c.get(c_data.as_mut_slice())?;
    println!("c = {:?}", c_data);

    Ok(())
}
