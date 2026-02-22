#[cfg(not(feature = "bench"))]
compile_error!("the bench feature is required for benchmarks, use `cargo bench --features bench`");

#[cfg(feature = "bench")]
mod benchmarks {
    use criterion::Criterion;
    pub use criterion::{criterion_group, criterion_main};
    use metatensor::Labels;

    const DATA_PATH: &str = "../../metatensor-core/tests/data.mts";

    pub fn partial_load_benchmarks(c: &mut Criterion) {
        let mut group = c.benchmark_group("load_partial");

        // Baseline: full load
        group.bench_function("full_load", |b| {
            b.iter(|| {
                std::hint::black_box(metatensor::io::load(DATA_PATH).unwrap());
            });
        });

        // load_partial (mmap-backed) with various filter combinations

        group.bench_function("no_filter", |b| {
            b.iter(|| {
                std::hint::black_box(
                    metatensor::io::load_partial(DATA_PATH, None, None, None).unwrap(),
                );
            });
        });

        let keys_sel = Labels::new(["o3_lambda"], &[[1]]);
        group.bench_function("key_filter", |b| {
            b.iter(|| {
                std::hint::black_box(
                    metatensor::io::load_partial(DATA_PATH, Some(&keys_sel), None, None).unwrap(),
                );
            });
        });

        let samples_sel = Labels::new(["system"], &[[0]]);
        group.bench_function("sample_filter", |b| {
            b.iter(|| {
                std::hint::black_box(
                    metatensor::io::load_partial(DATA_PATH, None, Some(&samples_sel), None).unwrap(),
                );
            });
        });

        let props_sel = Labels::new(["n"], &[[0]]);
        group.bench_function("property_filter", |b| {
            b.iter(|| {
                std::hint::black_box(
                    metatensor::io::load_partial(DATA_PATH, None, None, Some(&props_sel)).unwrap(),
                );
            });
        });

        group.bench_function("combined", |b| {
            b.iter(|| {
                std::hint::black_box(
                    metatensor::io::load_partial(
                        DATA_PATH, Some(&keys_sel), Some(&samples_sel), Some(&props_sel),
                    ).unwrap(),
                );
            });
        });

        group.finish();
    }
}

#[cfg(feature = "bench")]
benchmarks::criterion_group!(benches, benchmarks::partial_load_benchmarks);

#[cfg(feature = "bench")]
benchmarks::criterion_main!(benches);

#[cfg(not(feature = "bench"))]
fn main() {}
