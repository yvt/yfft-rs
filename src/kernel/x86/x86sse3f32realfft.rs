use super::utils::{if_compatible, AlignInfo, AlignReqKernel, AlignReqKernelWrapper};
use super::{Kernel, KernelParams, SliceAccessor};

use num_iter::range_step;
use packed_simd::{f32x4, u32x4};
use std::f32;
use std::mem;
use std::ptr::{read_unaligned, write_unaligned};

use aligned::AlignedVec;
use simdutils::{f32x4_bitxor, sse3_f32x4_complex_mul_riri};
use Num;

use super::x86sse1realfft::new_real_fft_coef_table;

/// Creates a real FFT post-processing or backward real FFT pre-processing kernel.
pub fn new_x86_sse3_f32_real_fft_pre_post_process_kernel<T>(
    len: usize,
    inverse: bool,
) -> Option<Box<Kernel<T>>>
where
    T: Num,
{
    if_compatible(|| {
        if len % 8 == 0 && len > 8 {
            Some(Box::new(AlignReqKernelWrapper::new(
                Sse3F32RealFFTPrePostProcessKernel::new(len, inverse),
            )) as Box<Kernel<f32>>)
        } else {
            None
        }
    })
}

#[derive(Debug)]
struct Sse3F32RealFFTPrePostProcessKernel {
    len: usize,
    table: [AlignedVec<f32>; 2],
    inverse: bool,
}

impl Sse3F32RealFFTPrePostProcessKernel {
    fn new(len: usize, inverse: bool) -> Self {
        Self {
            len,
            table: new_real_fft_coef_table(len, inverse),
            inverse,
        }
    }
}

impl AlignReqKernel<f32> for Sse3F32RealFFTPrePostProcessKernel {
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

        let conj_mask: f32x4 = unsafe { mem::transmute(u32x4::new(0, 0x80000000, 0, 0x80000000)) };

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
            let a1 = unsafe { read_unaligned(a_p1) };
            let a2 = unsafe { *a_p2 };
            let b1 = unsafe { read_unaligned(b_p1) };
            let b2 = unsafe { *b_p2 };

            let x1c = f32x4_bitxor(x1, conj_mask);
            let x2c = f32x4_bitxor(x2, conj_mask);
            let x1c = shuffle!(x1c, x1c, [2, 3, 4, 5]);
            let x2c = shuffle!(x2c, x2c, [2, 3, 4, 5]);

            let g1 = sse3_f32x4_complex_mul_riri(x1, a1) + sse3_f32x4_complex_mul_riri(x2c, b1);
            let g2 = sse3_f32x4_complex_mul_riri(x2, a2) + sse3_f32x4_complex_mul_riri(x1c, b2);

            unsafe {
                write_unaligned(cur1, g1);
                I::write(cur2, g2);
            }
        }
    }
    fn alignment_requirement(&self) -> usize {
        16
    }
}
