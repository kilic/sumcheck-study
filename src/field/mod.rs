pub use extension::*;
use std::fmt::Debug;

mod extension;
pub mod goldilocks;

pub trait FieldOps<Rhs = Self, Output = Self>:
    Copy
    + Send
    + Sync
    + core::ops::Add<Rhs, Output = Output>
    + core::ops::Sub<Rhs, Output = Output>
    + core::ops::Mul<Rhs, Output = Output>
{
}

pub trait FieldOpsAssigned<Rhs = Self>:
    Copy
    + Send
    + Sync
    + Sized
    + core::ops::AddAssign<Rhs>
    + core::ops::SubAssign<Rhs>
    + core::ops::MulAssign<Rhs>
{
}

// pub trait FieldOps: core::ops::Neg<Output = Self> + BaseFieldOps + BaseFieldOpsAssigned
// // + for<'a> BaseFieldOps<&'a Self, Self>
// // + for<'a> BaseFieldOpsAssigned<&'a Self>
// {
// }

pub trait Field:
    Sized
    + core::ops::Neg<Output = Self>
    + FieldOps
    + FieldOpsAssigned
    + Eq
    + Copy
    + Clone
    + Default
    + Send
    + Sync
    + Debug
    + 'static
    + From<bool>
    + From<u64>
    + From<u32>
    + From<u8>
    + core::iter::Sum
    + core::iter::Product
    + for<'a> core::iter::Sum<&'a Self>
    + for<'a> core::iter::Product<&'a Self>
{
    const ZERO: Self;
    const ONE: Self;
    const NEG_ONE: Self;
    const TWO: Self;
    const TWO_INV: Self;
    const NUM_BITS: usize;
    const NUM_BYTES: usize = (Self::NUM_BITS + 7) / 8;
    const GENERATOR: Self;

    fn rand(rng: impl rand::RngCore) -> Self;

    #[inline]
    fn double(&self) -> Self {
        *self + *self
    }

    #[inline]
    fn double_assign(&mut self) {
        *self += *self;
    }

    #[inline]
    fn square(&self) -> Self {
        *self * *self
    }

    #[inline]
    fn square_assign(&mut self) {
        *self *= *self;
    }

    #[inline]
    fn cube(&self) -> Self {
        *self * *self * *self
    }

    #[inline]
    fn cube_assign(&mut self) {
        *self = *self * *self * *self;
    }

    fn inverse(&self) -> Option<Self>;

    fn is_zero(&self) -> bool;

    fn is_one(&self) -> bool;

    fn pow<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::ONE;
        for e in exp.as_ref().iter().rev() {
            for i in (0..64).rev() {
                res = res.square();
                (((*e >> i) & 1) == 1).then(|| res.mul_assign(*self));
            }
        }
        res
    }

    #[inline]
    fn vanishing(&self, k: usize) -> Self {
        self.pow2(k) - Self::ONE
    }

    #[inline]
    fn shifted_vanishing(&self, k: usize, shift: Self) -> Self {
        self.pow2(k) - shift.pow2(k)
    }

    #[inline]
    fn pow2(self, n: usize) -> Self {
        (0..n).fold(self, |acc, _| acc.square())
    }

    fn powers(&self) -> impl Iterator<Item = Self> {
        std::iter::successors(Some(Field::ONE), |&acc| Some(acc * *self))
    }

    fn horner<'b, EF: Field, I>(&self, coeffs: I) -> Self
    where
        I: IntoIterator<Item = &'b EF>,
        I::IntoIter: DoubleEndedIterator,
        Self: std::ops::Add<EF, Output = Self>,
    {
        coeffs
            .into_iter()
            .rfold(Self::ZERO, |acc, &coeff| acc * *self + coeff)
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self>;
    fn from_uniform_bytes(bytes: &[u8]) -> Self;
    fn to_bytes(&self) -> Vec<u8>;
}

pub trait TwoAdicField: Field {
    const ROOT_OF_UNITY: Self;
    const TWO_ADICITY: usize;

    #[inline]
    fn omega(k: usize) -> Self {
        let t = Self::TWO_ADICITY.checked_sub(k).unwrap();
        Self::ROOT_OF_UNITY.pow2(t)
    }

    fn mul_subgroup(k: usize, shift: Self) -> Vec<Self> {
        Self::omega(k)
            .powers()
            .take(1 << k)
            .map(|v| v * shift)
            .collect()
    }
}
