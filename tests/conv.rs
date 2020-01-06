extern crate num_complex;
extern crate num_traits;
extern crate yfft;

use yfft::*;

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

// The number of patterns are reduced compared to that of `realfft.rs`
fn test_patterns<T: yfft::Num>(size: usize) -> Vec<Vec<T>> {
    let mut vec = Vec::new();
    vec.push(vec![T::zero(); size]);
    for x in 0..size {
        let mut vec2 = vec![T::zero(); size];
        vec2[x] = T::one();
        vec.push(vec2);
    }
    vec.push(
        (0..size)
            .map(|x| -> T { T::from(x).unwrap() })
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

fn conv<T: Num>() {
    let size = 32;

    let setup1: Setup<T> = Setup::new(&Options {
        input_data_order: DataOrder::Natural,
        output_data_order: DataOrder::Natural,
        input_data_format: DataFormat::Real,
        output_data_format: DataFormat::HalfComplex,
        len: size,
        inverse: false,
    })
    .unwrap();
    let setup2: Setup<T> = Setup::new(&Options {
        input_data_order: DataOrder::Natural,
        output_data_order: DataOrder::Natural,
        input_data_format: DataFormat::HalfComplex,
        output_data_format: DataFormat::Real,
        len: size,
        inverse: true,
    })
    .unwrap();

    let scale = T::from(2.0 / size as f64).unwrap();

    let mut env1 = Env::new(&setup1);
    let mut env2 = Env::new(&setup2);

    let patterns = test_patterns(size);
    for pat1 in patterns.iter() {
        for pat2 in patterns.iter() {
            let mut buf1 = pat1.clone();
            let mut buf2 = pat2.clone();
            env1.transform(&mut buf1);
            env1.transform(&mut buf2);
            spectrum_convolve(&mut buf1, &buf2);
            env2.transform(&mut buf1);
            cyclic_convolve(&mut buf2, pat1, pat2);

            for x in buf1.iter_mut() {
                *x = *x * scale;
            }

            assert_num_slice_approx_eq(&buf1, &buf2, T::from(1.0e-3).unwrap());
        }
    }
}

#[test]
fn conv_f32() {
    conv::<f32>();
}

#[test]
fn conv_f64() {
    conv::<f64>();
}

fn cyclic_convolve<T: Num>(out: &mut [T], in1: &[T], in2: &[T]) {
    for (i, out) in out.iter_mut().enumerate() {
        let mut sum = T::zero();
        for k in 0..in2.len() {
            sum += in1[(i + in1.len() - k) % in1.len()] * in2[k];
        }
        *out = sum;
    }
}

fn spectrum_convolve<T: Num>(buffer: &mut [T], ir_fq: &[T]) {
    buffer[0] = buffer[0] * ir_fq[0];
    buffer[1] = buffer[1] * ir_fq[1];
    for i in 1..buffer.len() / 2 {
        let (r1, i1) = (buffer[i * 2], buffer[i * 2 + 1]);
        let (r2, i2) = (ir_fq[i * 2], ir_fq[i * 2 + 1]);
        buffer[i * 2] = r1 * r2 - i1 * i2;
        buffer[i * 2 + 1] = r1 * i2 + r2 * i1;
    }
}
