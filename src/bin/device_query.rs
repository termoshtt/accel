extern crate accel;

use accel::device::Device;
use accel::error::Result;

fn device_query() -> Result<()> {
    for dev in Device::usables()? {
        println!("ID         = {:?}", dev);
        println!("name       = {:?}", dev.name()?);
        println!("FLOPS      = {:?}", dev.flops()?);
        println!("Capability = {:?}", dev.compute_capability()?);
    }
}

fn main() {
    device_query().unwrap();
}
