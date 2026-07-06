use metatensor::{EmptyArray, Labels, TensorBlock, TensorMap};


mod utils {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::Instant;

    use regex::Regex;

    #[derive(Clone, Copy)]
    pub struct BenchmarkResult {
        samples: usize,
        mean: f64,
        std: f64,
        min: f64,
        max: f64,
    }

    const MIN_SAMPLES: usize = 20;
    const MAX_TOTAL_TIME_S: f64 = 2.0;

    // if a single call takes less than this, we will use batching to get more
    // accurate timings
    const BENCHMARK_BATCH_THRESHOLD_S: f64 = 1e-6;
    const BATCH_SIZE: usize = 100;

    pub fn bench_function_with_setup<Setup, Func, Args>(
        mut func: Func,
        mut setup: Setup,
        n_samples: Option<usize>,
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

        // check if batching is needed by timing one extra call
        let use_batching = {
            let args = setup();
            let bench_start = Instant::now();
            func(args);
            bench_start.elapsed().as_secs_f64() < BENCHMARK_BATCH_THRESHOLD_S
        };

        let mut times = Vec::new();
        if let Some(n_samples) = n_samples {
            if use_batching {
                for _ in 0..n_samples {
                    let all_args: Vec<Args> = (0..BATCH_SIZE).map(|_| setup()).collect();
                    let bench_start = Instant::now();
                    for args in all_args {
                        func(args);
                    }
                    times.push(bench_start.elapsed().as_secs_f64() / BATCH_SIZE as f64);
                }
            } else {
                for _ in 0..n_samples {
                    let args = setup();
                    let start = Instant::now();
                    func(args);
                    times.push(start.elapsed().as_secs_f64());
                }
            }
        } else {
                if use_batching {
                let mut total_time = 0.0;
                let mut n = 0;
                while n < MIN_SAMPLES || total_time <= MAX_TOTAL_TIME_S {
                    let all_args: Vec<Args> = (0..BATCH_SIZE).map(|_| setup()).collect();
                    let bench_start = Instant::now();
                    for args in all_args {
                        func(args);
                    }
                    let elapsed = bench_start.elapsed().as_secs_f64();
                    times.push(elapsed / BATCH_SIZE as f64);
                    total_time += elapsed;
                    n += 1;
                }
            } else {
                let mut total_time = 0.0;
                let mut n = 0;
                while n < MIN_SAMPLES || total_time <= MAX_TOTAL_TIME_S {
                    let start = Instant::now();
                    let args = setup();
                    let bench_start = Instant::now();
                    func(args);
                    let elapsed = bench_start.elapsed().as_secs_f64();
                    times.push(elapsed);
                    total_time += start.elapsed().as_secs_f64();
                    n += 1;
                }
            }
        }

        remove_outliers(&mut times);

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
            samples: times.len(),
            mean,
            std: var.sqrt(),
            min: times.iter().copied().fold(f64::INFINITY, f64::min),
            max: times.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }

    pub fn bench_function<Func>(
        mut func: Func,
        n_samples: Option<usize>,
        n_warmup: Option<usize>,
    ) -> BenchmarkResult
    where
        Func: FnMut(),
    {
        bench_function_with_setup(
            |_| func(),
            || (),
            n_samples,
            n_warmup,
        )
    }

    fn remove_outliers(times: &mut Vec<f64>) {
        let original_len = times.len();
        if original_len < 4 {
            return;
        }

        let mut sorted = times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q1 = sorted[sorted.len() / 4];
        let q3 = sorted[sorted.len() * 3 / 4];
        let iqr = q3 - q1;
        let lower = q1 - 3.0 * iqr;
        let upper = q3 + 3.0 * iqr;

        times.retain(|&t| t >= lower && t <= upper);


        if times.len() < original_len * 5 / 10 {
            eprintln!(
                "error: more than 50% of samples were outliers ({} out of {})",
                original_len - times.len(),
                original_len
            );
            std::process::exit(1);
        }

        if times.len() < 1000 && original_len >= 1000 {
            eprintln!(
                "error: too many outliers, not enough samples left ({} out of {})",
                times.len(),
                original_len
            );
            std::process::exit(1);
        }
    }

    fn format_samples(count: usize) -> String {
        if count >= 1_000_000 {
            format!("{}M", count / 1_000_000)
        } else if count >= 1_000 {
            format!("{}k", count / 1_000)
        } else {
            format!("{}", count)
        }
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

    fn format_benchmark_result(
        results: &[(String, BenchmarkResult)],
        baseline: Option<&HashMap<String, BenchmarkResult>>,
    ) -> String {
        let header = if baseline.is_some() {
            format!(
                "{:<35} {:>8} {:>14} {:>14} {:>14} {:>14} {:>12}",
                "Function", "Samples", "Mean", "Std", "Min", "Max", "vs baseline"
            )
        } else {
            format!(
                "{:<35} {:>8} {:>14} {:>14} {:>14} {:>14}",
                "Function", "Samples", "Mean", "Std", "Min", "Max"
            )
        };
        let mut lines = Vec::new();
        lines.push(header.clone());
        lines.push("-".repeat(header.len()));

        for (name, result) in results {
            let (unit, scale) = guess_unit(result.mean);
            let mut line = format!(
                "{name:<35}  {samples:>7}       {mean:>6.2}{unit}       {std:>6.2}{unit}       {min:>6.2}{unit}       {max:>6.2}{unit}",
                samples = format_samples(result.samples),
                mean = result.mean * scale,
                std = result.std * scale,
                min = result.min * scale,
                max = result.max * scale,
            );

            if let Some(baseline) = baseline {
                if let Some(b) = baseline.get(name) {
                    let ratio = result.mean / b.mean;
                    line.push_str(&format!("       {ratio:>6.2}x"));
                } else {
                    line.push_str("           N/A");
                }
            }

            lines.push(line);
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

    fn format_json(
        results: &[(String, BenchmarkResult)],
        baseline: Option<&HashMap<String, BenchmarkResult>>,
    ) -> String {
        let mut entries = Vec::new();
        for (name, result) in results {
            let mut fields = format!(
                "\"samples\":{},\"mean\":{},\"std\":{},\"min\":{},\"max\":{}",
                result.samples,
                result.mean,
                result.std,
                result.min,
                result.max,
            );

            if let Some(baseline) = baseline && let Some(b) = baseline.get(name) {
                let ratio = result.mean / b.mean;
                fields.push_str(&format!(",\"vs_baseline\":{}", ratio));
            }

            entries.push(format!("\"{}\":{{{}}}", json_escape(name), fields));
        }
        format!("{{{}}}", entries.join(","))
    }

    pub type BenchFn = fn(Option<usize>, Option<usize>) -> BenchmarkResult;

    pub fn run_benchmarks(benchmarks: &[(&str, BenchFn)]) {
        let mut filters = Vec::new();
        let mut output_json = false;
        let mut list_only = false;
        let mut test_mode = false;
        let mut show_help = false;
        let mut save_baseline: Option<String> = None;
        let mut baseline_name: Option<String> = None;

        {
            let mut args = std::env::args().skip(1);
            while let Some(argument) = args.next() {
                match argument.as_str() {
                    "--json" => output_json = true,
                    "--list" => list_only = true,
                    "--test" => test_mode = true,
                    "--help" => show_help = true,
                    "--bench" => {
                        // ignore, since this is passed by Cargo when running benchmarks
                    }
                    _ if argument == "--save-baseline" => {
                        save_baseline = args.next();
                    }
                    _ if argument.starts_with("--save-baseline=") => {
                        save_baseline = Some(argument["--save-baseline=".len()..].to_string());
                    }
                    _ if argument == "--baseline" => {
                        baseline_name = args.next();
                    }
                    _ if argument.starts_with("--baseline=") => {
                        baseline_name = Some(argument["--baseline=".len()..].to_string());
                    }
                    _ => filters.push(argument),
                }
            }
        }

        if show_help {
            let program = std::env::args().next().unwrap_or_else(|| "bench".to_string());
            println!("Usage: {program} [OPTIONS] [REGEX...]");
            println!("Run benchmarks and print the results.");
            println!();
            println!("Options:");
            println!("  --list                     list all available benchmarks and exit");
            println!("  --json                     output the results as JSON instead of a table");
            println!("  --test                     run in test mode (no warmup, 1 iteration)");
            println!("  --save-baseline=NAME       save benchmark results as a baseline JSON file");
            println!("  --save-baseline NAME       same as above");
            println!("  --baseline=NAME            load baseline JSON and display speedup/slowdown");
            println!("  --baseline NAME            same as above");
            println!("  --help                     show this help message and exit");
            println!("  REGEX                      filter benchmarks to run (default: .*)");
            return;
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

        let (n_samples, n_warmup) = if test_mode {
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
            let benchmark = std::panic::catch_unwind(|| function(n_samples, n_warmup));
            if let Ok(benchmark) = benchmark {
                results.push((name.to_string(), benchmark));
            } else {
                eprintln!("error while running {name}: benchmark panicked");
                break;
            }
        }

        let baseline_dir = std::env::var("METATENSOR_BENCHMARK_BASELINE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(PathBuf::from))
                    .unwrap_or_default()
            });

        if let Err(e) = std::fs::create_dir_all(&baseline_dir) {
            eprintln!("error: could not create baseline directory {}: {}", baseline_dir.display(), e);
        }

        if let Some(name) = save_baseline {
            let file_name = format!("rs-{}.json", name);
            let path = baseline_dir.join(file_name);
            let content = format_json(&results, None);
            if let Err(e) = std::fs::write(&path, content + "\n") {
                eprintln!("error: could not write baseline file {}: {}", path.display(), e);
            }
        }

        let baseline = baseline_name.and_then(|name| {
            let file_name = format!("rs-{}.json", name);
            let path = baseline_dir.join(file_name);
            let content = std::fs::read_to_string(&path).ok()?;
            let parsed = json::parse(&content).ok()?;
            let mut map = HashMap::new();
            for (key, value) in parsed.entries() {
                let mean = value["mean"].as_f64()?;
                let std = value["std"].as_f64()?;
                let min = value["min"].as_f64()?;
                let max = value["max"].as_f64()?;
                let samples = value["samples"].as_usize().unwrap_or(0);
                map.insert(
                    key.to_string(),
                    BenchmarkResult { samples, mean, std, min, max },
                );
            }
            Some(map)
        });

        if output_json {
            println!("{}", format_json(&results, baseline.as_ref()));
        } else {
            println!("Benchmark results for Rust API:");
            println!("{}", format_benchmark_result(&results, baseline.as_ref()));
        }
    }
}

use utils::{BenchFn, BenchmarkResult};
use utils::{bench_function, bench_function_with_setup};

fn range_labels(name: &str, count: usize) -> Labels {
    let values: Vec<[i32; 1]> = (0..count).map(|i| [i as i32]).collect();
    Labels::new_assume_unique([name], values)
}

fn bench_labels_small(n_samples: Option<usize>, n_warmup: Option<usize>) -> BenchmarkResult {
    bench_function(
        || {
            let values = vec![[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]];
            std::hint::black_box(Labels::new(["a", "b", "c"], values));
        },
        n_samples,
        n_warmup,
    )
}

fn bench_labels_large(n_samples: Option<usize>, n_warmup: Option<usize>) -> BenchmarkResult {
    let mut values = Vec::with_capacity(100 * 100 * 100);
    for i in 0..100 {
        for j in 0..100 {
            for k in 0..100 {
                values.push([i, j, k]);
            }
        }
    }

    bench_function_with_setup(
        |v| {
            std::hint::black_box(Labels::new(["a", "b", "c"], v));
        },
        || values.clone(),
        n_samples,
        n_warmup,
    )
}

fn bench_labels_large_assume_unique(n_samples: Option<usize>, n_warmup: Option<usize>) -> BenchmarkResult {
    let mut values = Vec::with_capacity(100 * 100 * 100);
    for i in 0..100 {
        for j in 0..100 {
            for k in 0..100 {
                values.push([i, j, k]);
            }
        }
    }

    bench_function_with_setup(
        |values| {
            std::hint::black_box(Labels::new_assume_unique(["a", "b", "c"], values));
        },
        || values.clone(),
        n_samples,
        n_warmup,
    )
}

fn bench_tensor_block(n_samples: Option<usize>, n_warmup: Option<usize>) -> BenchmarkResult {
    let samples = range_labels("samples", 100);
    let properties = range_labels("properties", 100);

    bench_function(
        || {
            let values = EmptyArray::new(vec![100, 100]);
            std::hint::black_box(
                TensorBlock::new(values, &samples, &[], &properties)
                    .expect("failed to create TensorBlock"),
            );
        },
        n_samples,
        n_warmup,
    )
}

fn bench_tensor_block_large(n_samples: Option<usize>, n_warmup: Option<usize>) -> BenchmarkResult {
    let samples = range_labels("samples", 10_000);
    let components = vec![
        range_labels("component_1", 10),
        range_labels("component_2", 5),
        range_labels("component_3", 10),
    ];
    let properties = range_labels("properties", 100);

    bench_function(
        || {
            let values = EmptyArray::new(vec![10_000, 10, 5, 10, 100]);
            std::hint::black_box(
                TensorBlock::new(values, &samples, &components, &properties)
                    .expect("failed to create TensorBlock"),
            );
        },
        n_samples,
        n_warmup,
    )
}

fn bench_tensor_map(n_samples: Option<usize>, n_warmup: Option<usize>) -> BenchmarkResult {
    let n_blocks = 10;
    let keys = range_labels("key", n_blocks);

    let samples = range_labels("samples", 100);
    let properties = range_labels("properties", 100);

    bench_function_with_setup(
        |blocks: Vec<TensorBlock>| {
            std::hint::black_box(TensorMap::new(keys.clone(), blocks).expect("failed to create TensorMap"));
        },
        || {
            let mut blocks = Vec::with_capacity(n_blocks);
            for _ in 0..n_blocks {
                blocks.push(
                    TensorBlock::new(
                        EmptyArray::new(vec![100, 100]),
                        &samples,
                        &[],
                        &properties
                    ).expect("failed to create TensorBlock"),
                );
            }
            blocks
        },
        n_samples,
        n_warmup,
    )
}

fn bench_tensor_map_large(n_samples: Option<usize>, n_warmup: Option<usize>) -> BenchmarkResult {
    let n_blocks = 10_000;
    let keys = range_labels("key", n_blocks);

    let samples = range_labels("samples", 100);
    let properties = range_labels("properties", 100);

    bench_function_with_setup(
        |blocks: Vec<TensorBlock>| {
            std::hint::black_box(TensorMap::new(keys.clone(), blocks).expect("failed to create TensorMap"));
        },
        || {
            let mut blocks = Vec::with_capacity(n_blocks);
            for _ in 0..n_blocks {
                blocks.push(
                    TensorBlock::new(
                        EmptyArray::new(vec![100, 100]),
                        &samples,
                        &[],
                        &properties
                    ).expect("failed to create TensorBlock"),
                );
            }
            blocks
        },
        n_samples,
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
