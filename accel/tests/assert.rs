use accel::*;
use accel_derive::kernel;

#[kernel]
pub fn assert() {
    accel_core::assert_eq!(1 + 2, 4);
}

#[test]
fn sync() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();
    let result = assert(&ctx, 1, 4, &());
    assert!(result.is_err()); // assertion failed
    Ok(())
}

#[test]
fn stream() -> error::Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();
    let stream = Stream::new(&ctx);

    let module = assert::Module::new(&ctx)?;
    module.stream_launch(&stream, 1, 4, &())?; // lanch will succeed
    assert!(stream.sync().is_err()); // assertion failed is detected in next sync
    Ok(())
}
