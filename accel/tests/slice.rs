use accel::*;
use accel_derive::kernel;

#[kernel]
fn add_slice(a: &[f64], b: &[f64], c: &mut [f64]) {
    let i = accel_core::index() as usize;
    if (i as usize) < c.len() {
        c[i] = a[i] + b[i];
    }
}

#[test]
fn main() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context_auto()?;

    let n = 32;
    let mut a = DeviceMemory::<f64>::managed(&ctx, n)?;
    let mut b = DeviceMemory::<f64>::managed(&ctx, n)?;
    let mut c = DeviceMemory::<f64>::managed(&ctx, n)?;

    for i in 0..n {
        a[i] = i as f64;
        b[i] = 2.0 * i as f64;
    }
    println!("a = {:?}", a.as_slice());
    println!("b = {:?}", b.as_slice());

    let grid = Grid::x(1);
    let block = Block::x(n as u32);
    add_slice(
        &ctx,
        grid,
        block,
        &(a.as_slice(), b.as_slice(), c.as_slice_mut()),
    )?;

    println!("c = {:?}", c.as_slice());
    Ok(())
}
