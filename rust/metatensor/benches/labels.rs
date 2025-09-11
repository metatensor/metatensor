use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use metatensor::{LabelValue, LabelsBuilder};

fn bench_labels_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("LabelsBuilder");

    for size in [100, 1_000, 10_000] {
        // Pre-build a LabelsBuilder with a given number of entries.
        let mut builder = LabelsBuilder::new(vec!["a", "b", "c"]);
        for i in 0..size {
            builder.add(&[i, i, i]);
        }

        // Benchmark the standard `finish` method.
        group.bench_function(BenchmarkId::new("finish", size), |b| {
            b.iter_batched(
                || builder.clone(),
                |builder| std::hint::black_box(builder.finish()),
                BatchSize::LargeInput
            );
        });

        group.bench_function(BenchmarkId::new("finish_assume_unique", size), |b| {
            b.iter_batched(
                || builder.clone(),
                |builder| std::hint::black_box(builder.finish_assume_unique()),
                BatchSize::LargeInput
            );
        });
    }
    group.finish();
}

fn bench_labels_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("Labels");

    fn labels_builder(size: isize) -> LabelsBuilder {
        let mut builder = LabelsBuilder::new(vec!["a", "b", "c"]);
        for i in 0..size {
            builder.add(&[i, i, i]);
        }
        for i in 0..size {
            builder.add(&[-i, 2 * i, i + 42]);
        }
        for i in 0..size {
            builder.add(&[i % 10, i % 49, 3 * i + 9]);
        }
        builder
    }

    let labels = labels_builder(10_000).finish();
    // initialize the `positions` hash map
    let entry = [LabelValue::new(0), LabelValue::new(0), LabelValue::new(0)];
    labels.position(&entry);

    for n_lookups in [1, 100, 1000] {
        group.bench_function(BenchmarkId::new("lookup", n_lookups), |b| {
            b.iter(|| {
                for i in 0..n_lookups {
                    let p = labels.position(&[
                        LabelValue::new(i),
                        LabelValue::new(i % 42),
                        LabelValue::new(i + 44)
                    ]);
                    std::hint::black_box(p);
                }
            });
        });
    }

    for size in [1, 1000, 100_000] {
        group.bench_function(BenchmarkId::new("init_position", 3 * size), |b| {
            b.iter_batched(
                || labels_builder(size).finish(),
                |labels| std::hint::black_box(labels.position(&entry)),
                BatchSize::LargeInput
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_labels_creation, bench_labels_lookup);
criterion_main!(benches);
