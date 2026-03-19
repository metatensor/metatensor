#[cfg(not(feature = "bench"))]
compile_error!("the bench feature is required for bencharks, use `cargo bench --features bench`");

#[cfg(feature = "bench")]
mod benchmarks {
    use criterion::{BatchSize, BenchmarkId, Criterion};
    pub use criterion::{criterion_group, criterion_main};
    use metatensor::{Array, Labels, LabelsBuilder, TensorBlock, TensorMap};
    use metatensor::c_api::mts_array_t;
    use ndarray::ArcArray;

    fn tensor_map(n_blocks: usize, n_samples: usize, n_properties: usize) -> TensorMap {
        let mut keys = LabelsBuilder::new(vec!["key_1", "key_2"]);
        for i in 0..n_blocks {
            keys.add(&[i, 0]);
        }
        let keys = keys.finish();

        let mut blocks = Vec::new();
        for _ in 0..n_blocks {
            let mut samples = LabelsBuilder::new(vec!["samples"]);
            for i in 0..n_samples {
                samples.add(&[i]);
            }
            let samples = samples.finish();

            let mut components_builder = LabelsBuilder::new(vec!["components"]);
            components_builder.add(&[0]);
            let components = [components_builder.finish()];

            let mut properties = LabelsBuilder::new(vec!["properties"]);
            for i in 0..n_properties {
                properties.add(&[i]);
            }
            let properties = properties.finish();

            let shape = vec![samples.count(), components[0].count(), properties.count()];
            let data = ArcArray::from_elem(shape, 1.0);

            blocks.push(TensorBlock::new(data, &samples, &components, &properties).unwrap());
        }

        TensorMap::new(keys, blocks).unwrap()
    }

    fn make_fill_value(scalar: f64) -> mts_array_t {
        let data: Box<dyn Array> = Box::new(ArcArray::from_elem(vec![1], scalar));
        mts_array_t::from(data)
    }

    pub fn keys_to_samples(c: &mut Criterion) {
        let mut group = c.benchmark_group("TensorMap::keys_to_samples");
        // reduce sample size for faster benchmark execution during dev/test
        group.sample_size(10);

        for n_blocks in [1, 10, 100] {
            let tensor = tensor_map(n_blocks, 1000, 1000);
            let keys_to_move = Labels::empty(vec!["key_1"]);
            group.bench_function(BenchmarkId::new("n_blocks", n_blocks), |b| {
                b.iter_batched(
                    || tensor.try_clone().unwrap(),
                    |tensor| std::hint::black_box(tensor.keys_to_samples(&keys_to_move, make_fill_value(0.0), true)),
                    BatchSize::LargeInput,
                );
            });
        }
        group.finish();
    }

    pub fn keys_to_properties(c: &mut Criterion) {
        let mut group = c.benchmark_group("TensorMap::keys_to_properties");
        group.sample_size(10);

        for n_blocks in [1, 10, 100] {
            let tensor = tensor_map(n_blocks, 1000, 1000);
            let keys_to_move = Labels::empty(vec!["key_1"]);
            group.bench_function(BenchmarkId::new("n_blocks", n_blocks), |b| {
                b.iter_batched(
                    || tensor.try_clone().unwrap(),
                    |tensor| std::hint::black_box(tensor.keys_to_properties(&keys_to_move, make_fill_value(0.0), true)),
                    BatchSize::LargeInput,
                );
            });
        }
        group.finish();
    }
}


#[cfg(feature = "bench")]
benchmarks::criterion_group!(benches, benchmarks::keys_to_samples, benchmarks::keys_to_properties);

#[cfg(feature = "bench")]
benchmarks::criterion_main!(benches);

#[cfg(not(feature = "bench"))]
fn main() {}
