use accel::*;
use accel_derive::kernel;
use anyhow::Result;

#[kernel]
pub fn assert() {
    accel_core::assert_eq!(1 + 2, 4);
}

#[test]
fn main() -> Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();
    let result = assert(&ctx, 1, 4, &());
    assert!(result.is_err()); // assertion failed
    Ok(())
}
