use metatensor::{Labels, LabelsBuilder, TensorBlock, TensorMap};


mod utils {
    use std::time::Instant;

    use regex::Regex;

    #[derive(Clone, Copy)]
    pub struct BenchmarkResult {
        mean: f64,
        std: f64,
        min: f64,
        max: f64,
    }

    pub fn bench_function_with_setup<Setup, Func, Args>(
        mut func: Func,
        mut setup: Setup,
        n_iters: Option<usize>,
        n_warmup: Option<usize>,
    ) -> BenchmarkResult
    where
        Setup: FnMut() -> Args,
        Func: FnMut(Args),
    {
        for _ in 0..n_warmup.unwrap_or(5) {
            let args = setup();
            func(args);
        }

        let mut times = Vec::new();
        if let Some(n_iters) = n_iters {
            for _ in 0..n_iters {
                let args = setup();
                let start = Instant::now();
                func(args);
                times.push(start.elapsed().as_secs_f64());
            }
        } else {
            let mut total_time = 0.0;
            let mut n_iters = 0;

            const MAX_TOTAL_TIME_S: f64 = 3.0;
            const MIN_ITERS: usize = 20;
            const MAX_ITERS: usize = 1000;

            while n_iters < MIN_ITERS || (total_time <= MAX_TOTAL_TIME_S && n_iters <= MAX_ITERS) {
                let start = Instant::now();
                let args = setup();
                let bench_start = Instant::now();
                func(args);
                let elapsed = bench_start.elapsed().as_secs_f64();
                times.push(elapsed);
                total_time += start.elapsed().as_secs_f64();
                n_iters += 1;
            }
        }

        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let var = times
            .iter()
            .map(|time| {
                let diff = time - mean;
                diff * diff
            })
            .sum::<f64>()
            / times.len() as f64;

        BenchmarkResult {
            mean,
            std: var.sqrt(),
            min: times.iter().copied().fold(f64::INFINITY, f64::min),
            max: times.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }

    pub fn bench_function<Func>(
        mut func: Func,
        n_iters: Option<usize>,
        n_warmup: Option<usize>,
    ) -> BenchmarkResult
    where
        Func: FnMut(),
    {
        bench_function_with_setup(
            |_| func(),
            || (),
            n_iters,
            n_warmup,
        )
    }

    fn guess_unit(time: f64) -> (&'static str, f64) {
        if time < 1e-6 {
            ("ns", 1e9)
        } else if time < 1e-3 {
            ("us", 1e6)
        } else if time < 1.0 {
            ("ms", 1e3)
        } else {
            ("s", 1.0)
        }
    }

    fn format_benchmark_result(results: &[(String, BenchmarkResult)]) -> String {
        let header = format!(
            "{:<45} {:>12} {:>12} {:>12} {:>12}",
            "Function", "Mean", "Std", "Min", "Max"
        );
        let mut lines = Vec::new();
        lines.push(header.clone());
        lines.push("-".repeat(header.len()));

        for (name, result) in results {
            let (unit, scale) = guess_unit(result.mean);
            lines.push(format!(
                "{name:<45}     {mean:>6.2}{unit}     {std:>6.2}{unit}     {min:>6.2}{unit}     {max:>6.2}{unit}",
                mean = result.mean * scale,
                std = result.std * scale,
                min = result.min * scale,
                max = result.max * scale,
            ));
        }

        format!("\n{}", lines.join("\n"))
    }

    fn json_escape(value: &str) -> String {
        let mut escaped = String::with_capacity(value.len());
        for ch in value.chars() {
            match ch {
                '\\' => escaped.push_str("\\\\"),
                '"' => escaped.push_str("\\\""),
                _ => escaped.push(ch),
            }
        }
        escaped
    }

    fn format_json(results: &[(String, BenchmarkResult)]) -> String {
        let mut entries = Vec::new();
        for (name, result) in results {
            entries.push(format!(
                "\"{}\":{{\"mean\":{},\"std\":{},\"min\":{},\"max\":{}}}",
                json_escape(name),
                result.mean,
                result.std,
                result.min,
                result.max,
            ));
        }
        format!("{{{}}}", entries.join(","))
    }

    pub type BenchFn = fn(Option<usize>, Option<usize>) -> BenchmarkResult;

    pub fn run_benchmarks(benchmarks: &[(&str, BenchFn)]) {
        let mut filters = Vec::new();
        let mut output_json = false;
        let mut list_only = false;
        let mut test_mode = false;

        for argument in std::env::args().skip(1) {
            match argument.as_str() {
                "--json" => output_json = true,
                "--list" => list_only = true,
                "--test" => test_mode = true,
                "--bench" => {
                    // ignore, since this is passed by Cargo when running benchmarks
                }
                _ => filters.push(argument),
            }
        }

        if filters.is_empty() {
            filters.push(String::from(".*"));
        }

        if list_only {
            println!("Available benchmarks:");
            for (name, _) in benchmarks {
                println!(" - {name}");
            }
            return;
        }

        let regexes: Vec<Regex> = filters
            .iter()
            .map(|pattern| Regex::new(pattern).expect("invalid regex pattern"))
            .collect();

        let (n_iters, n_warmup) = if test_mode {
            (Some(1), Some(0))
        } else {
            (None, None)
        };

        let mut results: Vec<(String, BenchmarkResult)> = Vec::new();

        for (name, function) in benchmarks {
            if !regexes.iter().any(|regex| regex.is_match(name)) {
                continue;
            }

            eprintln!("running {name}...");
            let benchmark = std::panic::catch_unwind(|| function(n_iters, n_warmup));
            if let Ok(benchmark) = benchmark {
                results.push((name.to_string(), benchmark));
            } else {
                eprintln!("error while running {name}: benchmark panicked");
                break;
            }
        }

        if output_json {
            println!("{}", format_json(&results));
        } else {
            println!("Benchmark results for Rust API:");
            println!("{}", format_benchmark_result(&results));
        }
    }
}

use utils::{BenchFn, BenchmarkResult};
use utils::{bench_function, bench_function_with_setup};

fn range_labels(name: &str, count: usize) -> Labels {
    let mut builder = LabelsBuilder::new(vec![name]);
    for i in 0..count {
        builder.add(&[i]);
    }
    builder.finish_assume_unique()
}

fn bench_labels_small(n_iters: Option<usize>, n_warmup: Option<usize>) -> BenchmarkResult {
    let values = vec![[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]];
    bench_function(
        || {
            std::hint::black_box(Labels::new(["a", "b", "c"], &values));
        },
        n_iters,
        n_warmup,
    )
}

fn bench_labels_large(n_iters: Option<usize>, n_warmup: Option<usize>) -> BenchmarkResult {
    let mut values = Vec::with_capacity(100 * 100 * 100);
    for i in 0..100 {
        for j in 0..100 {
            for k in 0..100 {
                values.push([i, j, k]);
            }
        }
    }

    bench_function(
        || {
            std::hint::black_box(Labels::new(["a", "b", "c"], &values));
        },
        n_iters,
        n_warmup,
    )
}

fn bench_labels_large_assume_unique(n_iters: Option<usize>, n_warmup: Option<usize>) -> BenchmarkResult {
    let mut values = Vec::with_capacity(100 * 100 * 100);
    for i in 0..100 {
        for j in 0..100 {
            for k in 0..100 {
                values.push([i, j, k]);
            }
        }
    }

    bench_function(
        || {
            let mut builder = LabelsBuilder::new(vec!["a", "b", "c"]);
            for value in &values {
                builder.add(value);
            }
            std::hint::black_box(builder.finish_assume_unique());
        },
        n_iters,
        n_warmup,
    )
}

fn bench_tensor_block(n_iters: Option<usize>, n_warmup: Option<usize>) -> BenchmarkResult {
    let samples = range_labels("samples", 100);
    let properties = range_labels("properties", 100);

    bench_function_with_setup(
        |values| {
            std::hint::black_box(
                TensorBlock::new(values, &samples, &[], &properties)
                    .expect("failed to create TensorBlock"),
            );
        },
        || {
            ndarray::Array::from_shape_fn((100, 100), |(i, j)| ((i * 100 + j) % 97) as f64)
                .into_dyn()
        },
        n_iters,
        n_warmup,
    )
}

fn bench_tensor_block_large(n_iters: Option<usize>, n_warmup: Option<usize>) -> BenchmarkResult {
    let samples = range_labels("samples", 10_000);
    let components = vec![
        range_labels("component_1", 10),
        range_labels("component_2", 5),
        range_labels("component_3", 10),
    ];
    let properties = range_labels("properties", 100);

    bench_function_with_setup(
        |values| {
            std::hint::black_box(
                TensorBlock::new(values, &samples, &components, &properties)
                    .expect("failed to create TensorBlock"),
            );
        },
        || {
            ndarray::Array::from_elem((10_000, 10, 5, 10, 100), 1.0)
                .into_dyn()
        },
        n_iters,
        n_warmup,
    )
}

fn bench_tensor_map(n_iters: Option<usize>, n_warmup: Option<usize>) -> BenchmarkResult {
    let n_blocks = 10;
    let keys = range_labels("key", n_blocks);

    let samples = range_labels("samples", 100);
    let properties = range_labels("properties", 100);
    let values = ndarray::Array::from_shape_fn((100, 100), |(i, j)| ((i * 100 + j) % 101) as f64)
        .into_dyn();

    bench_function_with_setup(
        |blocks: Vec<TensorBlock>| {
            std::hint::black_box(TensorMap::new(keys.clone(), blocks).expect("failed to create TensorMap"));
        },
        || {
            let mut blocks = Vec::with_capacity(n_blocks);
            for _ in 0..n_blocks {
                blocks.push(
                    TensorBlock::new(values.clone(), &samples, &[], &properties)
                        .expect("failed to create TensorBlock"),
                );
            }
            blocks
        },
        n_iters,
        n_warmup,
    )
}

fn bench_tensor_map_large(n_iters: Option<usize>, n_warmup: Option<usize>) -> BenchmarkResult {
    let n_blocks = 10_000;
    let keys = range_labels("key", n_blocks);

    let samples = range_labels("samples", 100);
    let properties = range_labels("properties", 100);
    let values = ndarray::Array::from_shape_fn((100, 100), |(i, j)| ((i * 100 + j) % 101) as f64)
        .into_dyn();

    bench_function_with_setup(
        |blocks: Vec<TensorBlock>| {
            std::hint::black_box(TensorMap::new(keys.clone(), blocks).expect("failed to create TensorMap"));
        },
        || {
            let mut blocks = Vec::with_capacity(n_blocks);
            for _ in 0..n_blocks {
                blocks.push(
                    TensorBlock::new(values.clone(), &samples, &[], &properties)
                        .expect("failed to create TensorBlock"),
                );
            }
            blocks
        },
        n_iters,
        n_warmup,
    )
}


fn main() {
    let benchmarks: Vec<(&str, BenchFn)> = vec![
        ("Labels/small", bench_labels_small),
        ("Labels/large", bench_labels_large),
        ("Labels/large_assume_unique", bench_labels_large_assume_unique),
        ("TensorBlock/small", bench_tensor_block),
        ("TensorBlock/large", bench_tensor_block_large),
        ("TensorMap/small", bench_tensor_map),
        ("TensorMap/large", bench_tensor_map_large),
    ];
    utils::run_benchmarks(&benchmarks);
}
