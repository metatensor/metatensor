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

        // --- Small file benchmarks (data.mts, ~294KB) ---

        // Baseline: full load
        group.bench_function("full_load/data.mts", |b| {
            b.iter(|| {
                std::hint::black_box(metatensor::io::load(DATA_PATH).unwrap());
            });
        });

        // Partial: no filters (should be ~same as full load)
        group.bench_function("partial_no_filter/data.mts", |b| {
            b.iter(|| {
                std::hint::black_box(
                    metatensor::io::load_partial(DATA_PATH, None, None, None).unwrap(),
                );
            });
        });

        // Partial: key filter — select only o3_lambda=1
        let keys_sel = Labels::new(["o3_lambda"], &[[1]]);
        group.bench_function("partial_key_filter/data.mts", |b| {
            b.iter(|| {
                std::hint::black_box(
                    metatensor::io::load_partial(
                        DATA_PATH,
                        Some(&keys_sel),
                        None,
                        None,
                    )
                    .unwrap(),
                );
            });
        });

        // Partial: sample filter — select only system=0
        let samples_sel = Labels::new(["system"], &[[0]]);
        group.bench_function("partial_sample_filter/data.mts", |b| {
            b.iter(|| {
                std::hint::black_box(
                    metatensor::io::load_partial(
                        DATA_PATH,
                        None,
                        Some(&samples_sel),
                        None,
                    )
                    .unwrap(),
                );
            });
        });

        // Partial: property filter — select only n=0
        let props_sel = Labels::new(["n"], &[[0]]);
        group.bench_function("partial_property_filter/data.mts", |b| {
            b.iter(|| {
                std::hint::black_box(
                    metatensor::io::load_partial(
                        DATA_PATH,
                        None,
                        None,
                        Some(&props_sel),
                    )
                    .unwrap(),
                );
            });
        });

        // Partial: combined filter
        group.bench_function("partial_combined_filter/data.mts", |b| {
            b.iter(|| {
                std::hint::black_box(
                    metatensor::io::load_partial(
                        DATA_PATH,
                        Some(&keys_sel),
                        Some(&samples_sel),
                        Some(&props_sel),
                    )
                    .unwrap(),
                );
            });
        });

        // --- Comparison: full load + manual slice vs load_partial ---
        // This shows the cost of load() followed by discarding data
        group.bench_function("full_load_then_discard/data.mts", |b| {
            b.iter(|| {
                let tm = metatensor::io::load(DATA_PATH).unwrap();
                // Simulate the work of selecting keys after full load
                let _keys = tm.keys();
                let _count = _keys.count();
                std::hint::black_box(tm);
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
