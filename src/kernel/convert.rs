use super::{Kernel, KernelParams, SliceAccessor};

use Num;

/// Creates a kernel that converts from the `Real` format to `Complex` format.
pub fn new_real_to_complex_kernel<T>(len: usize) -> Box<Kernel<T>>
where
    T: Num,
{
    Box::new(RealToComplexKernel { len })
}

#[derive(Debug)]
struct RealToComplexKernel {
    len: usize,
}

impl<T> Kernel<T> for RealToComplexKernel
where
    T: Num,
{
    fn transform(&self, params: &mut KernelParams<T>) {
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..self.len * 2]) };
        for i in (0..self.len).rev() {
            data[i * 2] = data[i];
            data[i * 2 + 1] = T::zero();
        }
    }
}

/// Creates a kernel that converts from the `HalfComplex` format to `Complex` format.
pub fn new_half_complex_to_complex_kernel<T>(len: usize) -> Box<Kernel<T>>
where
    T: Num,
{
    assert!(len % 2 == 0);
    Box::new(HalfComplexToComplexKernel { len })
}

#[derive(Debug)]
struct HalfComplexToComplexKernel {
    len: usize,
}

impl<T> Kernel<T> for HalfComplexToComplexKernel
where
    T: Num,
{
    fn transform(&self, params: &mut KernelParams<T>) {
        let mut data = unsafe { SliceAccessor::new(&mut params.coefs[0..self.len * 2]) };
        for i in 1..self.len / 2 {
            data[(self.len - i) * 2] = data[i * 2];
            data[(self.len - i) * 2 + 1] = -data[i * 2 + 1];
        }
        data[self.len] = data[1];
        data[self.len + 1] = T::zero();
        data[1] = T::zero();
    }
}
