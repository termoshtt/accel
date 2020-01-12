use accel::*;
use accel_derive::kernel;

#[kernel]
pub fn add(a: &[f64], b: &[f64], c: &mut [f64]) {
    let i = accel_core::index() as usize;
    if (i as usize) < c.len() {
        c[i] = a[i] + b[i];
    }
}

fn main() -> anyhow::Result<()> {
    let n = 32;
    let mut a = UVec::new(n).unwrap();
    let mut b = UVec::new(n).unwrap();
    let mut c = UVec::new(n).unwrap();

    for i in 0..n {
        a[i] = i as f64;
        b[i] = 2.0 * i as f64;
    }
    println!("a = {:?}", a.as_slice());
    println!("b = {:?}", b.as_slice());

    let grid = Grid::x(1);
    let block = Block::x(n as u32);
    add(grid, block, a.as_slice(), b.as_slice(), c.as_slice_mut())?;

    device::sync()?;
    println!("c = {:?}", c.as_slice());
    Ok(())
}
