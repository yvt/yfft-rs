use super::{Kernel, KernelCreationParams, KernelParams, KernelType};
use std::any::Any;
use std::{fmt, ptr};

pub trait StaticParams: fmt::Debug + 'static + Sync + Send {
    fn inverse(&self) -> bool;
    fn kernel_type(&self) -> KernelType;
    fn check_param(&self, cparams: &KernelCreationParams) {
        assert_eq!(cparams.inverse, self.inverse());
        assert_eq!(cparams.kernel_type, self.kernel_type());
    }
}

#[derive(Debug)]
struct StaticParamsDitForward {}
impl StaticParams for StaticParamsDitForward {
    #[inline]
    fn inverse(&self) -> bool {
        false
    }
    #[inline]
    fn kernel_type(&self) -> KernelType {
        KernelType::Dit
    }
}

#[derive(Debug)]
struct StaticParamsDitBackward {}
impl StaticParams for StaticParamsDitBackward {
    #[inline]
    fn inverse(&self) -> bool {
        true
    }
    #[inline]
    fn kernel_type(&self) -> KernelType {
        KernelType::Dit
    }
}

#[derive(Debug)]
struct StaticParamsDifForward {}
impl StaticParams for StaticParamsDifForward {
    #[inline]
    fn inverse(&self) -> bool {
        false
    }
    #[inline]
    fn kernel_type(&self) -> KernelType {
        KernelType::Dif
    }
}

#[derive(Debug)]
struct StaticParamsDifBackward {}
impl StaticParams for StaticParamsDifBackward {
    #[inline]
    fn inverse(&self) -> bool {
        true
    }
    #[inline]
    fn kernel_type(&self) -> KernelType {
        KernelType::Dif
    }
}

/// Poor man's generic lambda
pub trait StaticParamsConsumer<TRet> {
    fn consume<T: StaticParams>(self, cparams: &KernelCreationParams, sparams: T) -> TRet;
}

pub fn branch_on_static_params<F, T>(cparams: &KernelCreationParams, f: F) -> T
where
    F: StaticParamsConsumer<T>,
{
    match (cparams.kernel_type, cparams.inverse) {
        (KernelType::Dit, false) => f.consume(cparams, StaticParamsDitForward {}),
        (KernelType::Dif, false) => f.consume(cparams, StaticParamsDifForward {}),
        (KernelType::Dit, true) => f.consume(cparams, StaticParamsDitBackward {}),
        (KernelType::Dif, true) => f.consume(cparams, StaticParamsDifBackward {}),
    }
}

pub fn if_compatible<TExpect, TRequired, F>(f: F) -> Option<TRequired>
where
    Option<TRequired>: Any,
    Option<TExpect>: Any,
    F: FnOnce() -> Option<TExpect>,
{
    let mut ret_cell = None;
    if let Some(ret) = (&mut ret_cell as &mut Any).downcast_mut() {
        *ret = f();
    }
    ret_cell
}

#[derive(Debug)]
pub struct AlignReqKernelWrapper<T>(T);

impl<T> AlignReqKernelWrapper<T> {
    pub fn new(x: T) -> Self {
        AlignReqKernelWrapper(x)
    }
}

impl<T: AlignReqKernel<S>, S> Kernel<S> for AlignReqKernelWrapper<T> {
    fn transform(&self, params: &mut KernelParams<S>) {
        let a_req = self.0.alignment_requirement();
        let addr = params.coefs.as_ptr() as usize;
        if (addr & (a_req - 1)) != 0 {
            self.0.transform::<AlignInfoUnaligned>(params);
        } else {
            self.0.transform::<AlignInfoAligned>(params);
        }
    }
    fn required_work_area_size(&self) -> usize {
        self.0.required_work_area_size()
    }
}

pub trait AlignReqKernel<T>: fmt::Debug + Sized + Sync + Send {
    fn transform<I: AlignInfo>(&self, params: &mut KernelParams<T>);
    fn required_work_area_size(&self) -> usize {
        0
    }
    fn alignment_requirement(&self) -> usize;
}

pub trait AlignInfo: Sized {
    fn is_aligned() -> bool;
    unsafe fn read<T>(p: *const T) -> T;
    unsafe fn write<T>(p: *mut T, value: T);
}

struct AlignInfoAligned;

impl AlignInfo for AlignInfoAligned {
    fn is_aligned() -> bool {
        true
    }
    unsafe fn read<T>(p: *const T) -> T {
        ptr::read(p)
    }
    unsafe fn write<T>(p: *mut T, value: T) {
        ptr::write(p, value)
    }
}

struct AlignInfoUnaligned;

impl AlignInfo for AlignInfoUnaligned {
    fn is_aligned() -> bool {
        false
    }
    unsafe fn read<T>(p: *const T) -> T {
        ptr::read_unaligned(p)
    }
    unsafe fn write<T>(p: *mut T, value: T) {
        ptr::write_unaligned(p, value)
    }
}
