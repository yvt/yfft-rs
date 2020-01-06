
//! Defines generic FFT kernels optimized for certain known radix values, but without any specific processor or
//! instruction set specific optimizations.
//!
//! Performances
//! ------------
//!
//! According to a benchmark result, this kernel runs about 10x slower than a commercial-level FFT library on a Skylake
//! machine.

use super::utils::{branch_on_static_params, StaticParams, StaticParamsConsumer};
use super::{Kernel, KernelCreationParams, KernelParams, KernelType, SliceAccessor};

use num_complex::Complex;
use num_iter::range_step;
use num_traits::{One, Zero};

use super::super::{mul_pos_i, Num};

use std::fmt::Debug;
use std::marker::PhantomData;

pub fn new_specialized_generic_kernel<T>(cparams: &KernelCreationParams) -> Option<Box<Kernel<T>>>
where
    T: Num,
{
    branch_on_static_params(
        cparams,
        Factory::<T> {
            phantom: PhantomData,
        },
    )
}

struct Factory<T> {
    phantom: PhantomData<T>,
}
impl<T: Num> StaticParamsConsumer<Option<Box<Kernel<T>>>> for Factory<T> {
    fn consume<TSParams>(
        self,
        cparams: &KernelCreationParams,
        sparams: TSParams,
    ) -> Option<Box<Kernel<T>>>
    where
        TSParams: StaticParams,
        T: Num,
    {
        let full_circle = if cparams.inverse { 2 } else { -2 };
        let twiddle_delta = Complex::new(
            Zero::zero(),
            T::from(cparams.size / cparams.radix / cparams.unit).unwrap()
                * T::from(full_circle).unwrap()
                * T::PI()
                / T::from(cparams.size).unwrap(),
        )
        .exp();

        match cparams.radix {
            2 => Some(Box::new(SpecializedGenericDitKernel::<
                T,
                SmallFFT2<T>,
                TSParams,
            > {
                cparams: *cparams,
                twiddle_delta: twiddle_delta,
                small_fft: PhantomData,
                sparams: sparams,
            })),
            4 => Some(Box::new(SpecializedGenericDitKernel::<
                T,
                SmallFFT4<T>,
                TSParams,
            > {
                cparams: *cparams,
                twiddle_delta: twiddle_delta,
                small_fft: PhantomData,
                sparams: sparams,
            })),
            _ => None,
        }
    }
}

trait SmallFFT<T>: Debug + Default + 'static + Sync + Send {
    fn radix() -> usize;
    fn load(&mut self, data: &SliceAccessor<&mut [T]>, offset: usize, stride: usize);
    fn twiddle(&mut self, c: Complex<T>);
    fn transform_forward(&mut self);
    fn transform_backward(&mut self);
    fn store(&self, data: &mut SliceAccessor<&mut [T]>, offset: usize, stride: usize);
}

#[derive(Debug, Clone, Copy, Default)]
struct SmallFFT2<T> {
    x1: Complex<T>,
    x2: Complex<T>,
}

impl<T: Num> SmallFFT<T> for SmallFFT2<T> {
    #[inline]
    fn radix() -> usize {
        2
    }
    #[inline]
    fn load(&mut self, data: &SliceAccessor<&mut [T]>, offset: usize, stride: usize) {
        self.x1.re = data[offset];
        self.x1.im = data[offset + 1];
        self.x2.re = data[offset + stride];
        self.x2.im = data[offset + stride + 1];
    }
    #[inline]
    fn twiddle(&mut self, c: Complex<T>) {
        self.x2 = self.x2 * c;
    }
    #[inline]
    fn transform_forward(&mut self) {
        let orig = *self;
        self.x1 = orig.x1 + orig.x2;
        self.x2 = orig.x1 - orig.x2;
    }
    #[inline]
    fn transform_backward(&mut self) {
        self.transform_forward();
    }
    #[inline]
    fn store(&self, data: &mut SliceAccessor<&mut [T]>, offset: usize, stride: usize) {
        data[offset] = self.x1.re;
        data[offset + 1] = self.x1.im;
        data[offset + stride] = self.x2.re;
        data[offset + stride + 1] = self.x2.im;
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct SmallFFT4<T> {
    x1: Complex<T>,
    x2: Complex<T>,
    x3: Complex<T>,
    x4: Complex<T>,
}

impl<T: Num> SmallFFT<T> for SmallFFT4<T> {
    #[inline]
    fn radix() -> usize {
        4
    }
    #[inline]
    fn load(&mut self, data: &SliceAccessor<&mut [T]>, offset: usize, stride: usize) {
        self.x1.re = data[offset];
        self.x1.im = data[offset + 1];
        self.x2.re = data[offset + stride];
        self.x2.im = data[offset + stride + 1];
        self.x3.re = data[offset + stride * 2];
        self.x3.im = data[offset + stride * 2 + 1];
        self.x4.re = data[offset + stride * 3];
        self.x4.im = data[offset + stride * 3 + 1];
    }
    #[inline]
    fn twiddle(&mut self, c: Complex<T>) {
        let c2 = c * c;
        self.x2 = self.x2 * c;
        self.x3 = self.x3 * c2;
        self.x4 = self.x4 * (c * c2);
    }
    #[inline]
    fn transform_forward(&mut self) {
        let t1 = self.x1 + self.x3;
        let t2 = self.x2 + self.x4;
        let t3 = self.x1 - self.x3;
        let t4 = self.x2 - self.x4;
        self.x1 = t1 + t2;
        self.x2 = t3 - mul_pos_i(t4);
        self.x3 = t1 - t2;
        self.x4 = t3 + mul_pos_i(t4);
    }
    #[inline]
    fn transform_backward(&mut self) {
        let t1 = self.x1 + self.x3;
        let t2 = self.x2 + self.x4;
        let t3 = self.x1 - self.x3;
        let t4 = self.x2 - self.x4;
        self.x1 = t1 + t2;
        self.x2 = t3 + mul_pos_i(t4);
        self.x3 = t1 - t2;
        self.x4 = t3 - mul_pos_i(t4);
    }
    #[inline]
    fn store(&self, data: &mut SliceAccessor<&mut [T]>, offset: usize, stride: usize) {
        data[offset] = self.x1.re;
        data[offset + 1] = self.x1.im;
        data[offset + stride] = self.x2.re;
        data[offset + stride + 1] = self.x2.im;
        data[offset + stride * 2] = self.x3.re;
        data[offset + stride * 2 + 1] = self.x3.im;
        data[offset + stride * 3] = self.x4.re;
        data[offset + stride * 3 + 1] = self.x4.im;
    }
}

#[derive(Debug)]
struct SpecializedGenericDitKernel<T, TSmallFFT, TSParams> {
    cparams: KernelCreationParams,
    twiddle_delta: Complex<T>,
    small_fft: PhantomData<TSmallFFT>,
    sparams: TSParams,
}

impl<T, TSmallFFT, TSParams> Kernel<T> for SpecializedGenericDitKernel<T, TSmallFFT, TSParams>
where
    T: Num,
    TSmallFFT: SmallFFT<T>,
    TSParams: StaticParams,
{
    fn transform(&self, params: &mut KernelParams<T>) {
        let cparams = &self.cparams;
        let sparams = &self.sparams;
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..cparams.size * 2]) };

        let twiddle_delta = self.twiddle_delta;
        let mut small_fft = TSmallFFT::default();

        let radix = TSmallFFT::radix();
        assert_eq!(TSmallFFT::radix(), cparams.radix);

        let pre_twiddle = sparams.kernel_type() == KernelType::Dit;
        let post_twiddle = sparams.kernel_type() == KernelType::Dif;

        for x in range_step(0, cparams.size, cparams.unit * radix) {
            let mut twiddle_1: Complex<T> = Complex::one();
            for y in 0..cparams.unit {
                small_fft.load(&data, (x + y) * 2, cparams.unit * 2);

                if pre_twiddle {
                    small_fft.twiddle(twiddle_1);
                }
                if sparams.inverse() {
                    small_fft.transform_backward();
                } else {
                    small_fft.transform_forward();
                }
                if post_twiddle {
                    small_fft.twiddle(twiddle_1);
                }

                small_fft.store(&mut data, (x + y) * 2, cparams.unit * 2);

                twiddle_1 = twiddle_1 * twiddle_delta;
            }
        }
    }
}
