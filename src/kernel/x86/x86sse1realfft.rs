use super::utils::{if_compatible, AlignInfo, AlignReqKernel, AlignReqKernelWrapper};
use super::{Kernel, KernelParams, SliceAccessor};

use num_iter::range_step;
use packed_simd::{f32x4, u32x4};
use std::f32;
use std::mem;
use std::ptr::{read_unaligned, write_unaligned};

use aligned::AlignedVec;
use simdutils::{f32x4_bitxor, f32x4_complex_mul_rrii};
use {mul_pos_i, Complex, Num};

/// Creates a real FFT post-processing or backward real FFT pre-processing kernel.
pub fn new_x86_sse_real_fft_pre_post_process_kernel<T>(
    len: usize,
    inverse: bool,
) -> Option<Box<Kernel<T>>>
where
    T: Num,
{
    if_compatible(|| {
        if len % 8 == 0 && len > 8 {
            Some(Box::new(AlignReqKernelWrapper::new(
                SseRealFFTPrePostProcessKernel::new(len, inverse),
            )) as Box<Kernel<f32>>)
        } else {
            None
        }
    })
}

pub(super) fn new_real_fft_coef_table(len: usize, inverse: bool) -> [AlignedVec<f32>; 2] {
    assert!(len % 2 == 0);
    let mut table_a = AlignedVec::with_capacity(len);
    let mut table_b = AlignedVec::with_capacity(len);
    for i in 0..(len / 2) {
        let c = Complex::new(0f32, (i as f32) * -f32::consts::PI / (len / 2) as f32).exp();

        let a = (Complex::new(1f32, 0f32) - mul_pos_i(c)) * 0.5f32;
        let b = (Complex::new(1f32, 0f32) + mul_pos_i(c)) * 0.5f32;
        if inverse {
            table_a.push(a.re);
            table_a.push(-a.im);
            table_b.push(b.re);
            table_b.push(-b.im);
        } else {
            table_a.push(a.re);
            table_a.push(a.im);
            table_b.push(b.re);
            table_b.push(b.im);
        }
    }
    [table_a, table_b]
}

#[derive(Debug)]
struct SseRealFFTPrePostProcessKernel {
    len: usize,
    table: [AlignedVec<f32>; 2],
    inverse: bool,
}

impl SseRealFFTPrePostProcessKernel {
    fn new(len: usize, inverse: bool) -> Self {
        Self {
            len,
            table: new_real_fft_coef_table(len, inverse),
            inverse,
        }
    }
}

impl AlignReqKernel<f32> for SseRealFFTPrePostProcessKernel {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<f32>) {
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..self.len]) };
        let table_a = unsafe { SliceAccessor::new(&self.table[0][..]) };
        let table_b = unsafe { SliceAccessor::new(&self.table[1][..]) };
        let len_2 = self.len / 2;
        if !self.inverse {
            let (x1, x2) = (data[0], data[1]);
            data[0] = x1 + x2;
            data[1] = x1 - x2;
        } else {
            let (x1, x2) = (data[0], data[1]);
            data[0] = (x1 + x2) * 0.5f32;
            data[1] = (x1 - x2) * 0.5f32;
        }

        let neg_mask: f32x4 = unsafe { mem::transmute(u32x4::new(0x80000000, 0x80000000, 0, 0)) };
        let conj_mask: f32x4 = unsafe { mem::transmute(u32x4::new(0, 0, 0x80000000, 0x80000000)) };
        for i in range_step(1, len_2 / 2, 2) {
            let cur1 = &mut data[i * 2] as *mut f32 as *mut f32x4;
            let cur2 = &mut data[(len_2 - i - 1) * 2] as *mut f32 as *mut f32x4;

            let a_p1 = &table_a[i * 2] as *const f32 as *const f32x4;
            let a_p2 = &table_a[(len_2 - i - 1) * 2] as *const f32 as *const f32x4;
            let b_p1 = &table_b[i * 2] as *const f32 as *const f32x4;
            let b_p2 = &table_b[(len_2 - i - 1) * 2] as *const f32 as *const f32x4;

            // riri
            let x1 = unsafe { read_unaligned(cur1) };
            let x2 = unsafe { I::read(cur2) };
            let a1i = unsafe { read_unaligned(a_p1) };
            let a2i = unsafe { *a_p2 };
            let b1i = unsafe { read_unaligned(b_p1) };
            let b2i = unsafe { *b_p2 };

            // riri to rrii
            let t1 = shuffle!(x1, x1, [0, 2, 5, 7]);
            let t2 = shuffle!(x2, x2, [0, 2, 5, 7]);
            let a1 = shuffle!(a1i, a1i, [0, 2, 5, 7]);
            let a2 = shuffle!(a2i, a2i, [0, 2, 5, 7]);
            let b1 = shuffle!(b1i, b1i, [0, 2, 5, 7]);
            let b2 = shuffle!(b2i, b2i, [0, 2, 5, 7]);

            let t1c = f32x4_bitxor(t1, conj_mask);
            let t2c = f32x4_bitxor(t2, conj_mask);
            let t1c = shuffle!(t1c, t1c, [1, 0, 7, 6]);
            let t2c = shuffle!(t2c, t2c, [1, 0, 7, 6]);

            let g1 = f32x4_complex_mul_rrii(t1, a1, neg_mask)
                + f32x4_complex_mul_rrii(t2c, b1, neg_mask);
            let g2 = f32x4_complex_mul_rrii(t2, a2, neg_mask)
                + f32x4_complex_mul_rrii(t1c, b2, neg_mask);

            // rrii to riri
            let y1 = shuffle!(g1, g1, [0, 2, 5, 7]);
            let y2 = shuffle!(g2, g2, [0, 2, 5, 7]);

            unsafe {
                write_unaligned(cur1, y1);
                I::write(cur2, y2);
            }
        }
    }
    fn alignment_requirement(&self) -> usize {
        16
    }
}
