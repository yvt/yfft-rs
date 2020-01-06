
extern crate test;

use self::test::Bencher;
use super::*;

// To convert the result into a BenchFFT MFLOPS, use the following formula:
//
//   mflops = 5000 N log2(N) / (time for one FFT in nanoseconds)
//            / 2 (for real FFTs)

fn run_single_benchmark(size: usize, b: &mut Bencher) {
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
    b.iter(move || {
        senv.transform(buf.as_mut_slice());
    })
}

#[bench]
fn simple_benchmark_00001(b: &mut Bencher) {
    run_single_benchmark(1, b);
}
#[bench]
fn simple_benchmark_00002(b: &mut Bencher) {
    run_single_benchmark(2, b);
}
#[bench]
fn simple_benchmark_00004(b: &mut Bencher) {
    run_single_benchmark(4, b);
}
#[bench]
fn simple_benchmark_00008(b: &mut Bencher) {
    run_single_benchmark(8, b);
}
#[bench]
fn simple_benchmark_00016(b: &mut Bencher) {
    run_single_benchmark(16, b);
}
#[bench]
fn simple_benchmark_00032(b: &mut Bencher) {
    run_single_benchmark(32, b);
}
#[bench]
fn simple_benchmark_00064(b: &mut Bencher) {
    run_single_benchmark(64, b);
}
#[bench]
fn simple_benchmark_00128(b: &mut Bencher) {
    run_single_benchmark(128, b);
}
#[bench]
fn simple_benchmark_00256(b: &mut Bencher) {
    run_single_benchmark(256, b);
}
#[bench]
fn simple_benchmark_00512(b: &mut Bencher) {
    run_single_benchmark(512, b);
}
#[bench]
fn simple_benchmark_01024(b: &mut Bencher) {
    run_single_benchmark(1024, b);
}
#[bench]
fn simple_benchmark_02048(b: &mut Bencher) {
    run_single_benchmark(2048, b);
}
#[bench]
fn simple_benchmark_04096(b: &mut Bencher) {
    run_single_benchmark(4096, b);
}
#[bench]
fn simple_benchmark_08192(b: &mut Bencher) {
    run_single_benchmark(8192, b);
}
#[bench]
fn simple_benchmark_16384(b: &mut Bencher) {
    run_single_benchmark(16384, b);
}

fn run_single_real_benchmark(size: usize, b: &mut Bencher) {
    let setup: Setup<f32> = Setup::new(&Options {
        input_data_order: DataOrder::Natural,
        output_data_order: DataOrder::Natural,
        input_data_format: DataFormat::Real,
        output_data_format: DataFormat::HalfComplex,
        len: size,
        inverse: false,
    })
    .unwrap();
    let mut senv = Env::new(&setup);
    let mut buf = vec![0f32; size];
    b.iter(move || {
        senv.transform(buf.as_mut_slice());
    })
}

#[bench]
fn simple_benchmark_real_00002(b: &mut Bencher) {
    run_single_real_benchmark(2, b);
}
#[bench]
fn simple_benchmark_real_00004(b: &mut Bencher) {
    run_single_real_benchmark(4, b);
}
#[bench]
fn simple_benchmark_real_00008(b: &mut Bencher) {
    run_single_real_benchmark(8, b);
}
#[bench]
fn simple_benchmark_real_00016(b: &mut Bencher) {
    run_single_real_benchmark(16, b);
}
#[bench]
fn simple_benchmark_real_00032(b: &mut Bencher) {
    run_single_real_benchmark(32, b);
}
#[bench]
fn simple_benchmark_real_00064(b: &mut Bencher) {
    run_single_real_benchmark(64, b);
}
#[bench]
fn simple_benchmark_real_00128(b: &mut Bencher) {
    run_single_real_benchmark(128, b);
}
#[bench]
fn simple_benchmark_real_00256(b: &mut Bencher) {
    run_single_real_benchmark(256, b);
}
#[bench]
fn simple_benchmark_real_00512(b: &mut Bencher) {
    run_single_real_benchmark(512, b);
}
#[bench]
fn simple_benchmark_real_01024(b: &mut Bencher) {
    run_single_real_benchmark(1024, b);
}
#[bench]
fn simple_benchmark_real_02048(b: &mut Bencher) {
    run_single_real_benchmark(2048, b);
}
#[bench]
fn simple_benchmark_real_04096(b: &mut Bencher) {
    run_single_real_benchmark(4096, b);
}
#[bench]
fn simple_benchmark_real_08192(b: &mut Bencher) {
    run_single_real_benchmark(8192, b);
}
#[bench]
fn simple_benchmark_real_16384(b: &mut Bencher) {
    run_single_real_benchmark(16384, b);
}

// Following tests doesn't include the bit revesal pass, so the result cannot be compared
// with other benchmarks

fn run_dif_benchmark(size: usize, b: &mut Bencher) {
    let setup: Setup<f32> = Setup::new(&Options {
        input_data_order: DataOrder::Natural,
        output_data_order: DataOrder::Swizzled,
        input_data_format: DataFormat::Complex,
        output_data_format: DataFormat::Complex,
        len: size,
        inverse: false,
    })
    .unwrap();
    let mut senv = Env::new(&setup);
    let mut buf = vec![0f32; size * 2];
    b.iter(move || {
        senv.transform(buf.as_mut_slice());
    })
}

#[bench]
fn dif_benchmark_00064(b: &mut Bencher) {
    run_dif_benchmark(64, b);
}
#[bench]
fn dif_benchmark_00256(b: &mut Bencher) {
    run_dif_benchmark(256, b);
}
#[bench]
fn dif_benchmark_00512(b: &mut Bencher) {
    run_dif_benchmark(512, b);
}
#[bench]
fn dif_benchmark_02048(b: &mut Bencher) {
    run_dif_benchmark(2048, b);
}
#[bench]
fn dif_benchmark_08192(b: &mut Bencher) {
    run_dif_benchmark(8192, b);
}

fn run_dit_benchmark(size: usize, b: &mut Bencher) {
    let setup: Setup<f32> = Setup::new(&Options {
        input_data_order: DataOrder::Swizzled,
        output_data_order: DataOrder::Natural,
        input_data_format: DataFormat::Complex,
        output_data_format: DataFormat::Complex,
        len: size,
        inverse: false,
    })
    .unwrap();
    let mut senv = Env::new(&setup);
    let mut buf = vec![0f32; size * 2];
    b.iter(move || {
        senv.transform(buf.as_mut_slice());
    })
}

#[bench]
fn dit_benchmark_00064(b: &mut Bencher) {
    run_dit_benchmark(64, b);
}
#[bench]
fn dit_benchmark_00256(b: &mut Bencher) {
    run_dit_benchmark(256, b);
}
#[bench]
fn dit_benchmark_00512(b: &mut Bencher) {
    run_dit_benchmark(512, b);
}
#[bench]
fn dit_benchmark_02048(b: &mut Bencher) {
    run_dit_benchmark(2048, b);
}
#[bench]
fn dit_benchmark_08192(b: &mut Bencher) {
    run_dit_benchmark(8192, b);
}
