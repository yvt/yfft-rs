use super::kernel::KernelParams;
use super::{Num, Setup};
use num_traits::Zero;
use std::borrow::Borrow;

/// Encapsulates the working area required for a transformation.
#[derive(Debug, Clone)]
pub struct Env<TNum, TSetupRef> {
    setup: TSetupRef,
    work_area: Vec<TNum>,
}

impl<TNum, TSetupRef> Env<TNum, TSetupRef>
where
    TNum: Num + 'static,
    TSetupRef: Borrow<Setup<TNum>>,
{
    pub fn new(setup: TSetupRef) -> Self {
        let work_area_size = setup.borrow().required_work_area_size();
        Env {
            setup: setup,
            work_area: vec![Zero::zero(); work_area_size],
        }
    }

    /// Transforms the supplied complex array `data` and writes the result to the same array (therefore this is an
    /// in-place operation).
    ///
    /// The array must be arranged in a interleaved order, in which the odd-th and even-th elements represent the real
    /// and imaginary components, respectively.
    pub fn transform(&mut self, data: &mut [TNum]) {
        let mut kernel_param = KernelParams {
            coefs: data,
            work_area: self.work_area.as_mut_slice(),
        };
        let setup = self.setup.borrow();
        // println!("{:#?}", setup);
        // println!("{:#?}", &unsafe{::std::mem::transmute::<&[TNum], &[::num_complex::Complex<TNum>]>(kernel_param.coefs)}[0 .. kernel_param.coefs.len() / 2]);
        for kernel in &setup.kernels {
            kernel.transform(&mut kernel_param);
            // println!("{:#?}", &unsafe{::std::mem::transmute::<&[TNum], &[::num_complex::Complex<TNum>]>(kernel_param.coefs)}[0 .. kernel_param.coefs.len() / 2]);
        }
    }
}
