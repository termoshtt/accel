use accel::*;
use accel_derive::kernel;
use anyhow::Result;

#[kernel]
pub fn assert() {
    accel_core::assert_eq!(1 + 2, 4);
}

fn main() -> Result<()> {
    let device = Device::nth(0)?;
    let ctx = device.create_context();
    let stream = Stream::new(&ctx);

    let module = assert::Module::new(&ctx)?;
    module.stream_launch(&stream, 1, 4, &())?; // lanch will succeed
    assert!(stream.sync().is_err()); // assertion failed is detected in next sync
    Ok(())
}
