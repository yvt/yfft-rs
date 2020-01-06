use super::utils::{if_compatible, AlignInfo, AlignReqKernel, AlignReqKernelWrapper};
use super::{Kernel, KernelParams, SliceAccessor};

use num_iter::range_step;
use packed_simd::{f32x8, u32x8};
use std::f32;
use std::mem;
use std::ptr::{read_unaligned, write_unaligned};

use aligned::AlignedVec;
use simdutils::{avx_f32x8_bitxor, avx_f32x8_complex_mul_riri};
use Num;

use super::x86sse1realfft::new_real_fft_coef_table;

/// Creates a real FFT post-processing or backward real FFT pre-processing kernel.
pub fn new_x86_avx_f32_real_fft_pre_post_process_kernel<T>(
    len: usize,
    inverse: bool,
) -> Option<Box<Kernel<T>>>
where
    T: Num,
{
    if_compatible(|| {
        if len % 16 == 0 && len > 16 {
            Some(Box::new(AlignReqKernelWrapper::new(
                AvxF32RealFFTPrePostProcessKernel::new(len, inverse),
            )) as Box<Kernel<f32>>)
        } else {
            None
        }
    })
}

#[derive(Debug)]
struct AvxF32RealFFTPrePostProcessKernel {
    len: usize,
    table: [AlignedVec<f32>; 2],
    inverse: bool,
}

impl AvxF32RealFFTPrePostProcessKernel {
    fn new(len: usize, inverse: bool) -> Self {
        Self {
            len,
            table: new_real_fft_coef_table(len, inverse),
            inverse,
        }
    }
}

impl AlignReqKernel<f32> for AvxF32RealFFTPrePostProcessKernel {
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

        let conj_mask: f32x8 = unsafe {
            mem::transmute(u32x8::new(
                0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000,
            ))
        };

        for i in range_step(1, len_2 / 2, 4) {
            let cur1 = &mut data[i * 2] as *mut f32 as *mut f32x8;
            let cur2 = &mut data[(len_2 - i - 3) * 2] as *mut f32 as *mut f32x8;

            let a_p1 = &table_a[i * 2] as *const f32 as *const f32x8;
            let a_p2 = &table_a[(len_2 - i - 3) * 2] as *const f32 as *const f32x8;
            let b_p1 = &table_b[i * 2] as *const f32 as *const f32x8;
            let b_p2 = &table_b[(len_2 - i - 3) * 2] as *const f32 as *const f32x8;

            // riri
            let x1 = unsafe { read_unaligned(cur1) };
            let x2 = unsafe { I::read(cur2) };
            let a1 = unsafe { read_unaligned(a_p1) };
            let a2 = unsafe { *a_p2 };
            let b1 = unsafe { read_unaligned(b_p1) };
            let b2 = unsafe { *b_p2 };

            let x1c = avx_f32x8_bitxor(x1, conj_mask);
            let x2c = avx_f32x8_bitxor(x2, conj_mask);
            let x1c = shuffle!(x1c, x1c, [6, 7, 4, 5, 2, 3, 0, 1]);
            let x2c = shuffle!(x2c, x2c, [6, 7, 4, 5, 2, 3, 0, 1]);

            let g1 = avx_f32x8_complex_mul_riri(x1, a1) + avx_f32x8_complex_mul_riri(x2c, b1);
            let g2 = avx_f32x8_complex_mul_riri(x2, a2) + avx_f32x8_complex_mul_riri(x1c, b2);

            unsafe {
                write_unaligned(cur1, g1);
                I::write(cur2, g2);
            }
        }
    }

    fn alignment_requirement(&self) -> usize {
        32
    }
}
