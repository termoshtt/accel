use accel::{device::Device, error::*};

fn device_query() -> Result<()> {
    for dev in Device::usables()? {
        println!("ID         = {:?}", dev);
        println!("name       = {}", dev.name()?);
        println!("FLOPS(DP)  = {:e}", dev.flops()?);
        println!("Capability = {:?}", dev.compute_capability()?);
    }
    Ok(())
}

fn main() {
    device_query().unwrap();
}
