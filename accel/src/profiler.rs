use crate::*;
use cuda::*;
use std::sync::Arc;

pub fn start(ctx: Arc<Context>) -> error::Result<()> {
    unsafe { contexted_call!(&ctx, cuProfilerStart) }
}

pub fn stop(ctx: Arc<Context>) -> error::Result<()> {
    unsafe { contexted_call!(&ctx, cuProfilerStop) }
}
