use crate::{Dual, HyperDual, Jet, Numeric};

pub trait Primal {
    fn to_f64(&self) -> f64;

    fn to_f32(&self) -> f32;
}

impl Primal for f64 {
    fn to_f64(&self) -> f64 {
        *self
    }

    fn to_f32(&self) -> f32 {
        *self as f32
    }
}

impl Primal for f32 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }

    fn to_f32(&self) -> f32 {
        *self
    }
}

impl<T: Numeric + Primal> Primal for Dual<T> {
    fn to_f64(&self) -> f64 {
        self.value.to_f64()
    }

    fn to_f32(&self) -> f32 {
        self.value.to_f32()
    }
}

impl<T: Numeric + Primal> Primal for HyperDual<T> {
    fn to_f64(&self) -> f64 {
        self.real.to_f64()
    }

    fn to_f32(&self) -> f32 {
        self.real.to_f32()
    }
}

impl<T: Numeric + Primal, const N: usize> Primal for Jet<T, N> {
    fn to_f64(&self) -> f64 {
        self.value().to_f64()
    }

    fn to_f32(&self) -> f32 {
        self.value().to_f32()
    }
}
