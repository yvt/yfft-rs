//! Defines Radix-2 FFT kernels optimized by using SSE instruction set.
//!
//! Performances
//! ------------
//!
//! According to a benchmark result, this kernel runs about 2-4x slower than a commercial-level FFT library (with
//! all optimizations and instruction sets including ones that this kernel doesn't support enabled) on a Skylake
//! machine.

use super::super::super::simdutils::{f32x4_bitxor, f32x4_complex_mul_rrii};
use super::utils::{
    branch_on_static_params, if_compatible, AlignInfo, AlignReqKernel, AlignReqKernelWrapper,
    StaticParams, StaticParamsConsumer,
};
use super::{Kernel, KernelCreationParams, KernelParams, KernelType, Num, SliceAccessor};

use num_complex::Complex;
use num_iter::range_step;

use packed_simd::f32x4;

use std::f32;

pub fn new_x86_sse_radix2_kernel<T>(cparams: &KernelCreationParams) -> Option<Box<Kernel<T>>>
where
    T: Num,
{
    if cparams.radix != 2 {
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
            unit if unit % 4 == 0 => Some(Box::new(AlignReqKernelWrapper::new(
                SseRadix2Kernel3::new(cparams, sparams),
            ))),
            unit if unit % 2 == 0 => Some(Box::new(AlignReqKernelWrapper::new(
                SseRadix2Kernel2::new(cparams, sparams),
            ))),
            1 => Some(Box::new(AlignReqKernelWrapper::new(SseRadix2Kernel1 {
                cparams: *cparams,
            }))),
            _ => None,
        }
    }
}

/// This Radix-2 kernel is specialized for the case where `unit == 1` and computes one small FFTs in a single iteration.
#[derive(Debug)]
struct SseRadix2Kernel1 {
    cparams: KernelCreationParams,
}

impl AlignReqKernel<f32> for SseRadix2Kernel1 {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<f32>) {
        let cparams = &self.cparams;
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..cparams.size * 2]) };

        assert_eq!(cparams.radix, 2);
        assert_eq!(cparams.unit, 1);

        let neg_mask_raw: [u32; 4] = [0, 0, 0x80000000, 0x80000000];
        let neg_mask = unsafe { *(&neg_mask_raw as *const u32 as *const f32x4) };

        for x in range_step(0, cparams.size * 2, 4) {
            let cur = &mut data[x] as *mut f32 as *mut f32x4;
            // t1a, t1b : Complex<f32> = X[x/2 .. x/2 + 2]
            let t1 = unsafe { I::read(cur) };
            // t2a, t2b = t1b, t1a
            let t2 = shuffle!(t1, t1, [2, 3, 4, 5]);
            // t3a, t3b = t1a, -t1b
            let t3 = f32x4_bitxor(t1, neg_mask);
            // t4a, t4b = t2a + t3a, t3b + t3b = t1a + t1b, t1a - t1b
            let t4 = t2 + t3;
            // Y[x/2 .. x/2 + 2] = t4a, t4b
            unsafe { I::write(cur, t4) };
        }
    }
    fn alignment_requirement(&self) -> usize {
        16
    }
}

/// This Radix-2 kernel computes two small FFTs in a single iteration.
#[derive(Debug)]
struct SseRadix2Kernel2<T> {
    cparams: KernelCreationParams,
    twiddles: Vec<f32x4>,
    sparams: T,
}

impl<T: StaticParams> SseRadix2Kernel2<T> {
    fn new(cparams: &KernelCreationParams, sparams: T) -> Self {
        sparams.check_param(cparams);
        assert_eq!(cparams.radix, 2);
        assert_eq!(cparams.unit % 2, 0);

        let full_circle = if cparams.inverse { 2f32 } else { -2f32 };
        let twiddles = range_step(0, cparams.unit, 2)
            .map(|i| {
                let c1 = Complex::new(
                    0f32,
                    full_circle * (i) as f32 / (cparams.radix * cparams.unit) as f32
                        * f32::consts::PI,
                )
                .exp();
                let c2 = Complex::new(
                    0f32,
                    full_circle * (i + 1) as f32 / (cparams.radix * cparams.unit) as f32
                        * f32::consts::PI,
                )
                .exp();
                // rrii format
                f32x4::new(c1.re, c2.re, c1.im, c2.im)
            })
            .collect();

        Self {
            cparams: *cparams,
            twiddles: twiddles,
            sparams: sparams,
        }
    }
}

impl<T: StaticParams> AlignReqKernel<f32> for SseRadix2Kernel2<T> {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<f32>) {
        // TODO: SSE3 version of this kernel
        // (SSE3 can perform complex multiplication in the RIRI format efficiently)
        let cparams = &self.cparams;
        let sparams = &self.sparams;
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..cparams.size * 2]) };

        let twiddles = unsafe { SliceAccessor::new(self.twiddles.as_slice()) };

        let neg_mask_raw: [u32; 4] = [0x80000000, 0x80000000, 0, 0];
        let neg_mask = unsafe { *(&neg_mask_raw as *const u32 as *const f32x4) };

        let pre_twiddle = sparams.kernel_type() == KernelType::Dit;
        let post_twiddle = sparams.kernel_type() == KernelType::Dif;

        for x in range_step(0, cparams.size * 2, cparams.unit * 4) {
            for y in 0..cparams.unit / 2 {
                let cur1 = &mut data[x + y * 4] as *mut f32 as *mut f32x4;
                let cur2 = &mut data[x + y * 4 + cparams.unit * 2] as *mut f32 as *mut f32x4;
                let twiddle_1 = twiddles[y];

                // x1a, x1b : Complex<f32> = X[x1/2 .. x1/2 + 2]
                // (x1a.r, x1a.i, x1b.r, x1b.i)
                let x1 = unsafe { I::read(cur1) };
                // y1a, y1b : Complex<f32> = X[x2/2 .. x2/2 + 2]
                // (y1a.r, y1a.i, y1b.r, y1b.i)
                let y1 = unsafe { I::read(cur2) };

                // apply twiddle factor
                // (y1a.r, y1b.r, y1a.i, y1b.i)
                let x2 = x1;
                let y2 = if pre_twiddle {
                    let t1 = shuffle!(y1, y1, [0, 2, 5, 7]); // riri to rrii
                    let t2 = f32x4_complex_mul_rrii(t1, twiddle_1, neg_mask);
                    shuffle!(t2, t2, [0, 2, 5, 7]) // rrii to riri
                } else {
                    y1
                };

                // perform size-2 FFT
                // (y3a.r, y3a.i, y3b.r, y3b.i)
                let x3 = x2 + y2;
                let y3 = x2 - y2;

                // apply twiddle factor
                // (y1a.r, y1b.r, y1a.i, y1b.i)
                let x4 = x3;
                let y4 = if post_twiddle {
                    let t1 = shuffle!(y3, y3, [0, 2, 5, 7]); // riri to rrii
                    let t2 = f32x4_complex_mul_rrii(t1, twiddle_1, neg_mask);
                    shuffle!(t2, t2, [0, 2, 5, 7]) // rrii to riri
                } else {
                    y3
                };

                unsafe { I::write(cur1, x4) };
                unsafe { I::write(cur2, y4) };
            }
        }
    }
    fn alignment_requirement(&self) -> usize {
        16
    }
}

/// This Radix-2 kernel computes four small FFTs in a single iteration.
#[derive(Debug)]
struct SseRadix2Kernel3<T> {
    cparams: KernelCreationParams,
    twiddles: Vec<f32x4>,
    sparams: T,
}

impl<T: StaticParams> SseRadix2Kernel3<T> {
    fn new(cparams: &KernelCreationParams, sparams: T) -> Self {
        sparams.check_param(cparams);
        assert_eq!(cparams.radix, 2);
        assert_eq!(cparams.unit % 4, 0);

        let full_circle = if cparams.inverse { 2f32 } else { -2f32 };
        let twiddles = range_step(0, cparams.unit, 2)
            .map(|i| {
                let k = i / 4 * 4;
                let c1 = Complex::new(
                    0f32,
                    full_circle * (k) as f32 / (cparams.radix * cparams.unit) as f32
                        * f32::consts::PI,
                )
                .exp();
                let c2 = Complex::new(
                    0f32,
                    full_circle * (k + 1) as f32 / (cparams.radix * cparams.unit) as f32
                        * f32::consts::PI,
                )
                .exp();
                let c3 = Complex::new(
                    0f32,
                    full_circle * (k + 2) as f32 / (cparams.radix * cparams.unit) as f32
                        * f32::consts::PI,
                )
                .exp();
                let c4 = Complex::new(
                    0f32,
                    full_circle * (k + 3) as f32 / (cparams.radix * cparams.unit) as f32
                        * f32::consts::PI,
                )
                .exp();
                // rrrr-iiii format
                // TODO: more efficient creation
                if i % 4 != 0 {
                    f32x4::new(c1.im, c2.im, c3.im, c4.im)
                } else {
                    f32x4::new(c1.re, c2.re, c3.re, c4.re)
                }
            })
            .collect();

        Self {
            cparams: *cparams,
            twiddles: twiddles,
            sparams: sparams,
        }
    }
}

impl<T: StaticParams> AlignReqKernel<f32> for SseRadix2Kernel3<T> {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<f32>) {
        let cparams = &self.cparams;
        let sparams = &self.sparams;
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..cparams.size * 2]) };

        let twiddles = unsafe { SliceAccessor::new(self.twiddles.as_slice()) };

        let pre_twiddle = sparams.kernel_type() == KernelType::Dit;
        let post_twiddle = sparams.kernel_type() == KernelType::Dif;

        for x in range_step(0, cparams.size * 2, cparams.unit * 4) {
            for y in 0..cparams.unit / 4 {
                let cur1a = &mut data[x + y * 8] as *mut f32 as *mut f32x4;
                let cur1b = &mut data[x + y * 8 + 4] as *mut f32 as *mut f32x4;
                let cur2a = &mut data[x + y * 8 + cparams.unit * 2] as *mut f32 as *mut f32x4;
                let cur2b = &mut data[x + y * 8 + cparams.unit * 2 + 4] as *mut f32 as *mut f32x4;
                let twiddle_r = twiddles[y * 2];
                let twiddle_i = twiddles[y * 2 + 1];

                let x1a = unsafe { I::read(cur1a) };
                let x1b = unsafe { I::read(cur1b) };
                let y1a = unsafe { I::read(cur2a) };
                let y1b = unsafe { I::read(cur2b) };

                // convert riri-riri to rrrr-iiii (shufps)
                let x2r = shuffle!(x1a, x1b, [0, 2, 4, 6]);
                let x2i = shuffle!(x1a, x1b, [1, 3, 5, 7]);
                let y2r = shuffle!(y1a, y1b, [0, 2, 4, 6]);
                let y2i = shuffle!(y1a, y1b, [1, 3, 5, 7]);

                // apply twiddle factor
                let x3r = x2r;
                let x3i = x2i;
                let y3r = if pre_twiddle {
                    y2r * twiddle_r - y2i * twiddle_i
                } else {
                    y2r
                };
                let y3i = if pre_twiddle {
                    y2r * twiddle_i + y2i * twiddle_r
                } else {
                    y2i
                };

                // perform size-2 FFT
                let x4r = x3r + y3r;
                let x4i = x3i + y3i;
                let y4r = x3r - y3r;
                let y4i = x3i - y3i;

                // apply twiddle factor
                let x5r = x4r;
                let x5i = x4i;
                let y5r = if post_twiddle {
                    y4r * twiddle_r - y4i * twiddle_i
                } else {
                    y4r
                };
                let y5i = if post_twiddle {
                    y4r * twiddle_i + y4i * twiddle_r
                } else {
                    y4i
                };

                // convert to rrrr-iiii to riri-riri (unpcklps/unpckups)
                let x6a = shuffle!(x5r, x5i, [0, 4, 1, 5]);
                let x6b = shuffle!(x5r, x5i, [2, 6, 3, 7]);
                let y6a = shuffle!(y5r, y5i, [0, 4, 1, 5]);
                let y6b = shuffle!(y5r, y5i, [2, 6, 3, 7]);

                unsafe { I::write(cur1a, x6a) };
                unsafe { I::write(cur1b, x6b) };
                unsafe { I::write(cur2a, y6a) };
                unsafe { I::write(cur2b, y6b) };
            }
        }
    }
    fn alignment_requirement(&self) -> usize {
        16
    }
}
