//! Defines Radix-4 single precision FFT kernels optimized by using SSE3 instruction set.
//!
//! SSE3 adds some instructions that are helpful in single precision complex arithmetics:
//!
//!  - `addsubps` - `a - conj(b)` when `a` and `b` are "riri" format
//!  - `haddps` - horizontial add
//!  - `hsubps` - horizontial sub
//!
//! Performances
//! ------------
//!
//! For small transforms ties with a commercial-level FFT library, but tends to be much slower for large transforms.

use super::super::super::simdutils::{f32x4_bitxor, sse3_f32x4_complex_mul_riri};
use super::utils::{
    branch_on_static_params, if_compatible, AlignInfo, AlignReqKernel, AlignReqKernelWrapper,
    StaticParams, StaticParamsConsumer,
};
use super::{Kernel, KernelCreationParams, KernelParams, KernelType, Num, SliceAccessor};

use num_complex::Complex;
use num_iter::range_step;

use packed_simd::f32x4;

use std::f32;

pub fn new_x86_sse3_f32_radix4_kernel<T>(cparams: &KernelCreationParams) -> Option<Box<Kernel<T>>>
where
    T: Num,
{
    if cparams.radix != 4 {
        return None;
    }

    if_compatible(|| branch_on_static_params(cparams, Factory {}))
}

struct Factory {}
impl StaticParamsConsumer<Option<Box<Kernel<f32>>>> for Factory {
    fn consume<T>(self, cparams: &KernelCreationParams, sparams: T) -> Option<Box<Kernel<f32>>>
    where
        T: StaticParams,
    {
        match cparams.unit {
            unit if unit % 4 == 0 => None,
            // some heuristics here... (we really need some sophiscated planning using run-time measurement, not heuristics)
            unit if unit % 2 == 0 && cparams.size <= 8192 => Some(Box::new(
                AlignReqKernelWrapper::new(Sse3Radix4Kernel2::new(cparams, sparams)),
            )),
            _ => None,
        }
    }
}

/// This Radix-4 kernel computes two small FFTs in a single iteration.
#[derive(Debug)]
struct Sse3Radix4Kernel2<T> {
    cparams: KernelCreationParams,
    twiddles: Vec<f32x4>,
    sparams: T,
}

impl<T: StaticParams> Sse3Radix4Kernel2<T> {
    fn new(cparams: &KernelCreationParams, sparams: T) -> Self {
        sparams.check_param(cparams);
        assert_eq!(cparams.radix, 4);
        assert_eq!(cparams.unit % 2, 0);

        let full_circle = if cparams.inverse { 2f32 } else { -2f32 };
        let mut twiddles = Vec::new();
        for i in range_step(0, cparams.unit, 2) {
            let c1 = Complex::new(
                0f32,
                full_circle * (i) as f32 / (cparams.radix * cparams.unit) as f32 * f32::consts::PI,
            )
            .exp();
            let c2 = Complex::new(
                0f32,
                full_circle * (i + 1) as f32 / (cparams.radix * cparams.unit) as f32
                    * f32::consts::PI,
            )
            .exp();
            // riri format
            twiddles.push(f32x4::new(c1.re, c1.im, c2.re, c2.im));

            let c12 = c1 * c1;
            let c22 = c2 * c2;
            twiddles.push(f32x4::new(c12.re, c12.im, c22.re, c22.im));

            let c13 = c12 * c1;
            let c23 = c22 * c2;
            twiddles.push(f32x4::new(c13.re, c13.im, c23.re, c23.im));
        }

        Self {
            cparams: *cparams,
            twiddles: twiddles,
            sparams: sparams,
        }
    }
}

impl<T: StaticParams> AlignReqKernel<f32> for Sse3Radix4Kernel2<T> {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<f32>) {
        let cparams = &self.cparams;
        let sparams = &self.sparams;
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..cparams.size * 2]) };

        let twiddles = unsafe { SliceAccessor::new(self.twiddles.as_slice()) };

        let neg_mask2_raw: [u32; 4] = [0x80000000, 0, 0x80000000, 0];
        let neg_mask2 = unsafe { *(&neg_mask2_raw as *const u32 as *const f32x4) };

        let pre_twiddle = sparams.kernel_type() == KernelType::Dit;
        let post_twiddle = sparams.kernel_type() == KernelType::Dif;

        for x in range_step(0, cparams.size * 2, cparams.unit * 8) {
            for y in 0..cparams.unit / 2 {
                let cur1 = &mut data[x + y * 4] as *mut f32 as *mut f32x4;
                let cur2 = &mut data[x + y * 4 + cparams.unit * 2] as *mut f32 as *mut f32x4;
                let cur3 = &mut data[x + y * 4 + cparams.unit * 4] as *mut f32 as *mut f32x4;
                let cur4 = &mut data[x + y * 4 + cparams.unit * 6] as *mut f32 as *mut f32x4;

                // riri format
                let twiddle_1 = twiddles[y * 3];
                let twiddle_2 = twiddles[y * 3 + 1];
                let twiddle_3 = twiddles[y * 3 + 2];

                // riri format
                let x1 = unsafe { I::read(cur1) };
                let y1 = unsafe { I::read(cur2) };
                let z1 = unsafe { I::read(cur3) };
                let w1 = unsafe { I::read(cur4) };

                // apply twiddle factor
                let x2 = x1;
                let y2 = if pre_twiddle {
                    sse3_f32x4_complex_mul_riri(y1, twiddle_1)
                } else {
                    y1
                };
                let z2 = if pre_twiddle {
                    sse3_f32x4_complex_mul_riri(z1, twiddle_2)
                } else {
                    z1
                };
                let w2 = if pre_twiddle {
                    sse3_f32x4_complex_mul_riri(w1, twiddle_3)
                } else {
                    w1
                };

                // perform size-4 FFT
                let x3 = x2 + z2;
                let y3 = y2 + w2;
                let z3 = x2 - z2;
                let w3t = y2 - w2;

                // w3 = w3t * i
                let w3 = f32x4_bitxor(shuffle!(w3t, w3t, [1, 0, 7, 6]), neg_mask2);

                let (x4, y4, z4, w4) = if sparams.inverse() {
                    (x3 + y3, z3 + w3, x3 - y3, z3 - w3)
                } else {
                    (x3 + y3, z3 - w3, x3 - y3, z3 + w3)
                };

                // apply twiddle factor
                let x5 = x4;
                let y5 = if post_twiddle {
                    sse3_f32x4_complex_mul_riri(y4, twiddle_1)
                } else {
                    y4
                };
                let z5 = if post_twiddle {
                    sse3_f32x4_complex_mul_riri(z4, twiddle_2)
                } else {
                    z4
                };
                let w5 = if post_twiddle {
                    sse3_f32x4_complex_mul_riri(w4, twiddle_3)
                } else {
                    w4
                };

                unsafe { I::write(cur1, x5) };
                unsafe { I::write(cur2, y5) };
                unsafe { I::write(cur3, z5) };
                unsafe { I::write(cur4, w5) };
            }
        }
    }
    fn alignment_requirement(&self) -> usize {
        16
    }
}
