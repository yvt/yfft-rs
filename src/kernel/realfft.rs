use super::{Kernel, KernelParams, SliceAccessor};

use {mul_pos_i, Complex, Num};

/// Creates a real FFT post-processing or backward real FFT pre-processing kernel.
pub fn new_real_fft_pre_post_process_kernel<T>(len: usize, inverse: bool) -> Box<Kernel<T>>
where
    T: Num,
{
    super::x86::new_x86_real_fft_pre_post_process_kernel(len, inverse).unwrap_or_else(|| {
        assert!(len % 2 == 0);
        Box::new(RealFFTPrePostProcessKernel {
            len,
            table: new_real_fft_coef_table(len, inverse),
            inverse,
        })
    })
}

pub(super) fn new_real_fft_coef_table<T>(len: usize, inverse: bool) -> Vec<T>
where
    T: Num,
{
    assert!(len % 2 == 0);
    let mut table = Vec::with_capacity(len * 2);
    let half = T::from(0.5).unwrap();
    for i in 0..(len / 2) {
        let c = Complex::new(
            T::zero(),
            T::from(i).unwrap() * -T::PI() / T::from(len / 2).unwrap(),
        )
        .exp();

        let a = (Complex::new(T::one(), T::zero()) - mul_pos_i(c)) * half;
        let b = (Complex::new(T::one(), T::zero()) + mul_pos_i(c)) * half;
        if inverse {
            table.push(a.re);
            table.push(-a.im);
            table.push(b.re);
            table.push(-b.im);
        } else {
            table.push(a.re);
            table.push(a.im);
            table.push(b.re);
            table.push(b.im);
        }
    }
    table
}

#[derive(Debug)]
struct RealFFTPrePostProcessKernel<T> {
    len: usize,
    table: Vec<T>,
    inverse: bool,
}

impl<T> Kernel<T> for RealFFTPrePostProcessKernel<T>
where
    T: Num,
{
    fn transform(&self, params: &mut KernelParams<T>) {
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..self.len]) };
        let table = unsafe { SliceAccessor::new(&self.table[..]) };
        let len_2 = self.len / 2;
        if !self.inverse {
            // A(0) = (1 - j) / 2, B(0) = (1 + j) / 2
            // A(k) = (1 + j) / 2, B(k) = (1 - j) / 2
            // x A(0) + conj(x) B(0) = Re(x) + Im(x)
            // x A(k) + conj(x) B(k) = Re(x) - Im(x)
            // Store G(0) as X_r(0)
            // Store G(N/2) as X_i(0)
            // data[1] = data[0] - data[1];
            let (x1, x2) = (data[0], data[1]);
            data[0] = x1 + x2;
            data[1] = x1 - x2;
        } else {
            // A(0) = (1 + j) / 2, B(0) = (1 - j) / 2
            // A(k) = (1 - j) / 2, B(k) = (1 + j) / 2
            // Re(x) A(0) + Im(x) B(0) = (Re(x) + Im(x)) / 2 + j(Re(x) - Im(x)) /2
            let (x1, x2) = (data[0], data[1]);
            data[0] = (x1 + x2) * T::from(0.5).unwrap();
            data[1] = (x1 - x2) * T::from(0.5).unwrap();
        }
        for i in 1..len_2 / 2 + 1 {
            let a1r = table[i * 4];
            let b1r = table[i * 4 + 2];
            let a2r = table[(len_2 - i) * 4];
            let b2r = table[(len_2 - i) * 4 + 2];
            let x1r = data[i * 2];
            let x2r = data[(len_2 - i) * 2];

            let a1i = table[i * 4 + 1];
            let b1i = table[i * 4 + 3];
            let a2i = table[(len_2 - i) * 4 + 1];
            let b2i = table[(len_2 - i) * 4 + 3];
            let x1i = data[i * 2 + 1];
            let x2i = data[(len_2 - i) * 2 + 1];

            let a1 = Complex::new(a1r, a1i);
            let b1 = Complex::new(b1r, b1i);
            let a2 = Complex::new(a2r, a2i);
            let b2 = Complex::new(b2r, b2i);
            let x1 = Complex::new(x1r, x1i);
            let x2 = Complex::new(x2r, x2i);

            let g1 = x1 * a1 + x2.conj() * b1;
            let g2 = x2 * a2 + x1.conj() * b2;

            data[i * 2] = g1.re;
            data[i * 2 + 1] = g1.im;
            data[(len_2 - i) * 2] = g2.re;
            data[(len_2 - i) * 2 + 1] = g2.im;
        }
    }
}
