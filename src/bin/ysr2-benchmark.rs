extern crate yfft;
use std::time;
use yfft::{DataFormat, DataOrder, Env, Options, Setup};

fn duration_to_secs(d: time::Duration) -> f64 {
    d.as_secs() as f64 + d.subsec_nanos() as f64 * 1.0e-9
}

fn estimate_unit_size<T: FnMut()>(mut x: T) -> u64 {
    // Estimate the appropriate iteration count (so the overhead of
    // `Instant::now()` wouldn't affect the result)
    let mut unit_size = 1u64;
    loop {
        let start = time::Instant::now();
        for _ in 0..unit_size {
            x();
        }
        let dur = duration_to_secs(start.elapsed());
        if dur > 0.5 {
            break;
        } else {
            unit_size *= 2;
        }
    }
    unit_size
}

/// Perform a benchmark on a given function.
///
/// Returns the CPU time required for each iteration.
#[inline(never)]
fn benchmark_single<T: FnMut()>(mut x: T, unit_size: u64) -> f64 {
    // Run the benchmark.
    let start = time::Instant::now();
    let mut total_iter = 0;

    while start.elapsed().as_secs() < 1 {
        for _ in 0..unit_size {
            x();
        }
        total_iter += unit_size;
    }

    // Compute the single iteration time
    duration_to_secs(start.elapsed()) / total_iter as f64
}

/// Perform a benchmark on a given function for multiple times.
///
/// Returns the CPU time required for each iteration as well as its standard
/// deviation.
fn benchmark<T: FnMut()>(mut x: T) -> (f64, f64) {
    let unit_size = estimate_unit_size(&mut x);

    let mut total = 0.0;
    let mut total_sq = 0.0;
    let count = 5usize;

    for _ in 0..count {
        let t = benchmark_single(&mut x, unit_size);
        total += t;
        total_sq += t * t;
    }

    total /= count as f64;
    total_sq /= count as f64;
    let variance = total_sq - total * total;

    (total, variance.sqrt())
}

fn run_single_benchmark(size: usize) {
    let setup: Setup<f32> = Setup::new(&Options {
        input_data_order: DataOrder::Natural,
        output_data_order: DataOrder::Natural,
        input_data_format: DataFormat::Complex,
        output_data_format: DataFormat::Complex,
        len: size,
        inverse: false,
    })
    .unwrap();
    let mut senv = Env::new(&setup);
    let mut buf = vec![0f32; size * 2];
    let (iter_time, sd) = benchmark(move || {
        senv.transform(buf.as_mut_slice());
    });
    let size_f = size as f64;
    let num_fops = size_f * size_f.log2() * 5.0;
    let mflops = num_fops / iter_time * 1.0e-6;
    println!(
        "cplx-to-cplx, N = {: >5}, t = {: >9.2}, sd = {: >8.2},  mflops = {: >9.2}",
        size,
        iter_time * 1.0e9,
        sd * 1.0e9,
        mflops
    );
}

fn main() {
    println!("Running benchmark...");
    for i in 0..15 {
        run_single_benchmark(1 << i);
    }
}
