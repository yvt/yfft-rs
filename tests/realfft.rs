extern crate num_complex;
extern crate num_traits;
extern crate yfft;

use num_traits::One;

use yfft::*;

// TODO: test all kernels --- currently, only the kernels for the highest possible ISA are tested

fn assert_num_slice_approx_eq<T: yfft::Num>(got: &[T], expected: &[T], releps: T) {
    assert_eq!(got.len(), expected.len());
    // We can't use `Iterator::max()` because T doesn't implement Ord
    let maxabs = expected
        .iter()
        .map(|x| x.abs())
        .fold(T::zero() / T::zero(), |x, y| x.max(y))
        + T::from(0.01).unwrap();
    let eps = maxabs * releps;
    for i in 0..got.len() {
        let a = got[i];
        let b = expected[i];
        if (a - b).abs() > eps {
            assert!(
                (a - b).abs() < eps,
                "assertion failed: `got almost equal to expected` \
                 (got: `{:?}`, expected: `{:?}`, diff=`{:?}`)",
                got,
                expected,
                (a - b).abs()
            );
        }
    }
}

// thanks to the linearity of DFT, we only need as many test cases as the DFT size
// (unless some buggy code breaks it)
fn test_patterns<T: yfft::Num>(size: usize) -> Vec<Vec<T>> {
    let mut vec = Vec::new();
    vec.push(vec![T::zero(); size]);
    for x in 0..size {
        let mut vec2 = vec![T::zero(); size];
        vec2[x] = One::one();
        vec.push(vec2);
    }
    vec.push(
        (0..size)
            .map(|x| -> T { T::from(x).unwrap() })
            .collect::<Vec<T>>(),
    );
    vec.push(
        (0..size)
            .map(|x| -> T { T::from(x * 3 + 7).unwrap() })
            .collect::<Vec<T>>(),
    );
    vec.push(
        (0..size)
            .map(|x| -> T { T::from(-(x as isize)).unwrap() })
            .collect::<Vec<T>>(),
    );
    vec.push(
        (0..size)
            .map(|x| -> T { T::from((x * 3 + 7) & 0xf).unwrap() })
            .collect::<Vec<T>>(),
    );
    vec.push(
        (0..size)
            .map(|x| -> T { T::from((x * 3 + 7) ^ (x * 7 + 3) ^ (x >> 1)).unwrap() })
            .collect::<Vec<T>>(),
    );

    vec
}

// assumes complex FFT is okay
fn fft_real_forward<T: Num>() {
    for size_ref in &[1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 40, 49, 64, 128] {
        let size = *size_ref;

        // real FFT
        let setup1: Setup<T> = Setup::new(&Options {
            input_data_order: DataOrder::Natural,
            output_data_order: DataOrder::Natural,
            input_data_format: DataFormat::Real,
            output_data_format: DataFormat::Complex,
            len: size * 2,
            inverse: false,
        })
        .unwrap();

        // complex FFT
        let setup2: Setup<T> = Setup::new(&Options {
            input_data_order: DataOrder::Natural,
            output_data_order: DataOrder::Natural,
            input_data_format: DataFormat::Complex,
            output_data_format: DataFormat::Complex,
            len: size * 2,
            inverse: false,
        })
        .unwrap();

        let mut env1 = Env::new(&setup1);
        let mut env2 = Env::new(&setup2);

        let mut result1 = vec![T::zero(); size * 4];
        let mut result2 = vec![T::zero(); size * 4];
        for pat in test_patterns::<T>(size * 2) {
            // real FFT
            result1[0..size * 2].copy_from_slice(pat.as_slice());
            for i in size * 2..size * 4 {
                result1[i] = T::zero();
            }
            env1.transform(result1.as_mut_slice());

            // complex FFT
            for i in 0..size * 2 {
                result2[i * 2] = pat[i];
                result2[i * 2 + 1] = T::zero();
            }
            env2.transform(result2.as_mut_slice());

            assert_num_slice_approx_eq(
                result1.as_slice(),
                result2.as_slice(),
                T::from(1.0e-3).unwrap(),
            );
        }
    }
}

#[test]
fn fft_real_forward_f32() {
    fft_real_forward::<f32>();
}

#[test]
fn fft_real_forward_f64() {
    fft_real_forward::<f64>();
}

// assumes complex FFT is okay
fn fft_real_backward<T: Num>() {
    for size_ref in &[1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 40, 49, 64, 128] {
        let size = *size_ref;

        // real FFT
        let setup1: Setup<T> = Setup::new(&Options {
            input_data_order: DataOrder::Natural,
            output_data_order: DataOrder::Natural,
            input_data_format: DataFormat::HalfComplex,
            output_data_format: DataFormat::Complex,
            len: size * 2,
            inverse: true,
        })
        .unwrap();

        // complex FFT
        let setup2: Setup<T> = Setup::new(&Options {
            input_data_order: DataOrder::Natural,
            output_data_order: DataOrder::Natural,
            input_data_format: DataFormat::Complex,
            output_data_format: DataFormat::Complex,
            len: size * 2,
            inverse: true,
        })
        .unwrap();

        let mut env1 = Env::new(&setup1);
        let mut env2 = Env::new(&setup2);

        let mut result1 = vec![T::zero(); size * 4];
        let mut result2 = vec![T::zero(); size * 4];
        for pat in test_patterns::<T>(size * 2) {
            // real FFT
            result1[0..size * 2].copy_from_slice(pat.as_slice());
            for i in size * 2..size * 4 {
                result1[i] = T::zero();
            }
            env1.transform(result1.as_mut_slice());

            // complex FFT
            result2[0..size * 2].copy_from_slice(pat.as_slice());
            for i in 1..size * 2 {
                result2[(size * 2 - i) * 2] = result2[i * 2];
                result2[(size * 2 - i) * 2 + 1] = -result2[i * 2 + 1];
            }
            result2[size * 2] = result2[1];
            result2[size * 2 + 1] = T::zero();
            result2[1] = T::zero();
            for e in result2.iter_mut() {
                *e = *e * T::from(0.5f32).unwrap();
            }
            env2.transform(result2.as_mut_slice());

            assert_num_slice_approx_eq(
                result1.as_slice(),
                result2.as_slice(),
                T::from(1.0e-3).unwrap(),
            );
        }
    }
}

#[test]
fn fft_real_backward_f32() {
    fft_real_backward::<f32>();
}

#[test]
fn fft_real_backward_f64() {
    fft_real_backward::<f64>();
}

fn fft_roundtrip_real<T: Num>() {
    for size_ref in &[1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 40, 49, 64, 128] {
        let size = *size_ref;

        let setup1: Setup<T> = Setup::new(&Options {
            input_data_order: DataOrder::Natural,
            output_data_order: DataOrder::Natural,
            input_data_format: DataFormat::Real,
            output_data_format: DataFormat::HalfComplex,
            len: size * 2,
            inverse: false,
        })
        .unwrap();
        let setup2: Setup<T> = Setup::new(&Options {
            input_data_order: DataOrder::Natural,
            output_data_order: DataOrder::Natural,
            input_data_format: DataFormat::HalfComplex,
            output_data_format: DataFormat::Real,
            len: size * 2,
            inverse: true,
        })
        .unwrap();

        let mut env1 = Env::new(&setup1);
        let mut env2 = Env::new(&setup2);

        let factor = T::one() / T::from(size).unwrap();

        let mut result = vec![T::zero(); size * 2];
        for pat in test_patterns::<T>(size * 2) {
            result.copy_from_slice(pat.as_slice());
            env1.transform(result.as_mut_slice());
            env2.transform(result.as_mut_slice());

            for e in &mut result {
                *e = *e * factor;
            }

            assert_num_slice_approx_eq(result.as_slice(), pat.as_slice(), T::from(1.0e-3).unwrap());
        }
    }
}

#[test]
fn fft_roundtrip_real_f32() {
    fft_roundtrip_real::<f32>();
}

#[test]
fn fft_roundtrip_real_f64() {
    fft_roundtrip_real::<f64>();
}
