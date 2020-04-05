use accel::*;
use accel_derive::kernel;

#[kernel]
unsafe fn add(a: *const f64, b: *const f64, c: *mut f64, n: usize) {
    let i = accel_core::index();
    if (i as usize) < n {
        *c.offset(i) = *a.offset(i) + *b.offset(i);
    }
}

#[test]
fn main() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();

    let n = 32;
    let mut a = DeviceMemory::<f64>::new(&ctx, n);
    let mut b = DeviceMemory::<f64>::new(&ctx, n);
    let mut c = DeviceMemory::<f64>::new(&ctx, n);

    for i in 0..n {
        a[i] = i as f64;
        b[i] = 2.0 * i as f64;
    }
    println!("a = {:?}", a.as_slice());
    println!("b = {:?}", b.as_slice());

    add(&ctx, 1, n, &(&a.as_ptr(), &b.as_ptr(), &c.as_mut_ptr(), &n)).expect("Kernel call failed");

    println!("c = {:?}", c.as_slice());
    Ok(())
}

#[test]
fn show_ptx_string() {
    // PTX assembler code is embedded as `add::PTX_STR`
    println!("{}", add::PTX_STR);
}
