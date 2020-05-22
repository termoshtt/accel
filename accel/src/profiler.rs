//! Profiling GPU kernels and host CUDA API calls

use crate::*;
use cuda::*;

/// RAII handler for nvprof profiling
///
/// - Profiling starts by `Profiler::start`, and stops by `Drop` of `Profiler`.
/// - Unified memory profiling is not supported. You must add an option `--unified-memory-profiling off` to `nvprof` command.
///   ```shell
///   $ nvprof --unified-memory-profiling off ./target/release/examples/add
///   ```
/// - You will find more options at [nvprof user's guide](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview)
pub struct Profiler {
    ctx: Context,
}

impl Drop for Profiler {
    fn drop(&mut self) {
        if let Err(e) = unsafe { contexted_call!(&self.ctx, cuProfilerStop) } {
            log::error!("Failed to stop profiling: {:?}", e);
        }
    }
}

impl Profiler {
    pub fn start(ctx: Context) -> Self {
        unsafe { contexted_call!(&ctx, cuProfilerStart) }.expect("Profiler has already started");
        Self { ctx }
    }
}
