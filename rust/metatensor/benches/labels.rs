use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use metatensor::LabelsBuilder;

fn bench_finish_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("LabelsBuilder");

    for size in [100, 1_000, 10_000].iter() {
        // Pre-build a LabelsBuilder with a given number of entries.
        let mut builder = LabelsBuilder::new(vec!["a", "b", "c"]);
        for i in 0..*size {
            builder.add(&[i, i, i]);
        }

        // Benchmark the standard `finish` method.
        // It consumes the builder, so clone it inside the benchmark loop.
        group.bench_with_input(BenchmarkId::new("finish", size), &builder, |b, builder| {
            // `b.iter` runs the closure multiple times to get a reliable measurement.
            b.iter(|| {
                let b = builder.clone();
                black_box(b.finish());
            });
        });

        group.bench_with_input(
            BenchmarkId::new("finish_assume_unique", size),
            &builder,
            |b, builder| {
                b.iter(|| {
                    let b = builder.clone();
                    black_box(b.finish_assume_unique());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_finish_methods);
criterion_main!(benches);
