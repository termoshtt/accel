use accel::*;
use criterion::*;

fn host_to_device(c: &mut Criterion) {
    let device = Device::nth(0).unwrap();
    let ctx = device.create_context();

    let mut group = c.benchmark_group("host_to_device");
    for &n in &[100, 1000, 10_000, 100_000] {
        {
            let host = vec![0_u32; n];
            let mut dev = DeviceMemory::zeros(ctx.clone(), n);
            group.bench_with_input(BenchmarkId::new("direct_vec", n), &n, |b, _| {
                b.iter(|| {
                    for i in 0..n {
                        dev[i] = host[i];
                    }
                })
            });
            group.bench_with_input(BenchmarkId::new("memcpy_vec", n), &n, |b, _| {
                b.iter(|| {
                    dev.copy_from(host.as_slice());
                })
            });
        }

        {
            let host = PageLockedMemory::<u32>::zeros(ctx.clone(), n);
            let mut dev = DeviceMemory::zeros(ctx.clone(), n);
            group.bench_with_input(BenchmarkId::new("direct_page_locked", n), &n, |b, _| {
                b.iter(|| {
                    for i in 0..n {
                        dev[i] = host[i];
                    }
                })
            });
            group.bench_with_input(BenchmarkId::new("memcpy_page_locked", n), &n, |b, _| {
                b.iter(|| {
                    dev.copy_from(host.as_slice());
                })
            });
        }

        {
            let mut vec = vec![0_u32; n];
            let host = RegisteredMemory::new(ctx.clone(), &mut vec);
            let mut dev = DeviceMemory::zeros(ctx.clone(), n);
            group.bench_with_input(BenchmarkId::new("direct_registered", n), &n, |b, _| {
                b.iter(|| {
                    for i in 0..n {
                        dev[i] = host[i];
                    }
                })
            });
            group.bench_with_input(BenchmarkId::new("memcpy_registered", n), &n, |b, _| {
                b.iter(|| {
                    dev.copy_from(host.as_slice());
                })
            });
        }
    }
}

criterion_group!(benches, host_to_device);
criterion_main!(benches);
