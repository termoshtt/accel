use accel::*;

#[kernel]
fn f(a: &i32, b: &mut i32) {
    accel_core::println!("&a = {:p}", a);
    accel_core::println!("&b = {:p}", b);
    *b = *a;
}

#[test]
fn copy() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();

    let a = 1_i32;
    let mut b = 0_i32;

    println!("&a = {:p}", &a);
    println!("&b = {:p}", &b);

    f(&ctx, 1, 1, (&a, &mut b))?;

    assert_eq!(a, b);
    Ok(())
}
