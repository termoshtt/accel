use accel::*;
use criterion::*;

fn h2d(c: &mut Criterion) {
    let device = Device::nth(0).unwrap();
    let context = device.create_context();
    let mut group = c.benchmark_group("h2d");

    macro_rules! impl_HtoD {
        ($host:expr, $id:expr) => {
            let host = $host;
            let n = host.len();
            let mut dev = DeviceMemory::zeros(&context, n);
            group.bench_with_input(
                BenchmarkId::new(&format!("direct_{}", $id), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        for i in 0..n {
                            dev[i] = host[i];
                        }
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("memcpy_{}", $id), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        dev.copy_from(&host);
                    })
                },
            );
        };
    }

    for &n in &[1000, 10_000, 100_000] {
        // impl_HtoD!(vec![0_u32; n], "vec");
        impl_HtoD!(PageLockedMemory::<u32>::zeros(&context, n), "page_locked");
        let mut vec_tmp = vec![0_u32; n];
        impl_HtoD!(RegisteredMemory::new(&context, &mut vec_tmp), "registered");
    }
}

fn d2h(c: &mut Criterion) {
    let device = Device::nth(0).unwrap();
    let context = device.create_context();
    let mut group = c.benchmark_group("d2h");

    macro_rules! impl_DtoH {
        ($host:expr, $id:expr) => {
            let mut host = $host;
            let n = host.len();
            let dev = DeviceMemory::zeros(&context, n);
            group.bench_with_input(
                BenchmarkId::new(&format!("direct_{}", $id), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        for i in 0..n {
                            host[i] = dev[i];
                        }
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("memcpy_{}", $id), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        host.copy_from(&dev);
                    })
                },
            );
        };
    }

    for &n in &[1000, 10_000, 100_000] {
        impl_DtoH!(vec![0_u32; n], "vec");
        impl_DtoH!(PageLockedMemory::<u32>::zeros(&context, n), "page_locked");
        let mut vec_tmp = vec![0_u32; n];
        impl_DtoH!(RegisteredMemory::new(&context, &mut vec_tmp), "registered");
    }
}

criterion_group!(benches, h2d, d2h);
criterion_main!(benches);
