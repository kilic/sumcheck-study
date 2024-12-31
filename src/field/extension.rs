use super::{Field, FieldOps, FieldOpsAssigned};
use itertools::Itertools;
use rand::{distributions::Standard, prelude::Distribution};

pub trait Extended<const E: usize, T = [Self; E]>: Field {
    const GENERATOR: T;
    const NON_RESIDUE: Self;
    #[inline]
    fn mul_by_nonresidue(&self) -> Self {
        Self::NON_RESIDUE * *self
    }
}

pub trait ExtField<F>: Field + FieldOps<F, Self> + FieldOpsAssigned<F> + From<F> {
    fn as_slice(&self) -> &[F];
    fn set0(&mut self, e: F);
}

#[derive(Debug, Clone, PartialEq, Eq, Copy, PartialOrd, Ord)]
pub struct Ext<const E: usize, F: Extended<E>>([F; E]);

impl<const E: usize, F: Extended<E>> FieldOps for Ext<E, F> {}
impl<const E: usize, F: Extended<E>> FieldOpsAssigned<Ext<E, F>> for Ext<E, F> {}
impl<const E: usize, F: Extended<E>> FieldOps<F, Ext<E, F>> for Ext<E, F> {}
impl<const E: usize, F: Extended<E>> FieldOpsAssigned<F> for Ext<E, F> {}

impl<const E: usize, F: Extended<E>> Default for Ext<E, F> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const E: usize, F: Extended<E>> Ext<E, F> {
    pub(crate) const fn from_base(e: F) -> Self {
        let mut inner = [F::ZERO; E];
        inner[0] = e;
        Self(inner)
    }
}

impl<const E: usize, F: Extended<E>> std::ops::Index<usize> for Ext<E, F> {
    type Output = F;

    fn index(&self, index: usize) -> &F {
        self.0.index(index)
    }
}

impl<const E: usize, F: Extended<E>> std::ops::IndexMut<usize> for Ext<E, F> {
    fn index_mut(&mut self, index: usize) -> &mut F {
        self.0.index_mut(index)
    }
}

impl<const E: usize, F: Extended<E>> From<F> for Ext<E, F> {
    fn from(e: F) -> Self {
        std::iter::once(e)
            .chain(std::iter::repeat(F::ZERO))
            .take(E)
            .collect_vec()
            .into()
    }
}

impl<const E: usize, F: Extended<E>> From<[F; E]> for Ext<E, F> {
    fn from(inner: [F; E]) -> Self {
        Self(inner)
    }
}

impl<const E: usize, F: Extended<E>> From<Vec<F>> for Ext<E, F> {
    fn from(inner: Vec<F>) -> Self {
        Self(inner.try_into().unwrap())
    }
}

impl<const E: usize, F: Extended<E>> FromIterator<F> for Ext<E, F> {
    fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
        iter.into_iter().collect_vec().into()
    }
}

impl<'a, const E: usize, F: Extended<E>> Iterator for &'a Ext<E, F> {
    type Item = &'a F;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter().next()
    }
}

impl<const E: usize, F: Extended<E>> From<&[F]> for Ext<E, F> {
    fn from(inner: &[F]) -> Self {
        Ext(inner.to_vec().try_into().unwrap())
    }
}

impl<const E: usize, F: Extended<E>> From<u64> for Ext<E, F> {
    fn from(val: u64) -> Self {
        let val: F = val.into();
        val.into()
    }
}

impl<const E: usize, F: Extended<E>> From<u32> for Ext<E, F> {
    fn from(val: u32) -> Self {
        let val: F = val.into();
        val.into()
    }
}

impl<const E: usize, F: Extended<E>> From<u8> for Ext<E, F> {
    fn from(val: u8) -> Self {
        let val: F = val.into();
        val.into()
    }
}

impl<const E: usize, F: Extended<E>> From<bool> for Ext<E, F> {
    fn from(val: bool) -> Self {
        let val: F = val.into();
        val.into()
    }
}

impl<const E: usize, F: Extended<E>> ExtField<F> for Ext<E, F> {
    fn as_slice(&self) -> &[F] {
        &self.0
    }
    fn set0(&mut self, e: F) {
        self[0] = e;
    }
}

impl<const E: usize, F: Extended<E>> Field for Ext<E, F> {
    const ZERO: Self = Self::from_base(F::ZERO);
    const ONE: Self = Self::from_base(F::ONE);
    const NEG_ONE: Self = Self::from_base(F::NEG_ONE);
    const TWO: Self = Self::from_base(F::TWO);
    const TWO_INV: Self = Self::from_base(F::TWO_INV);
    const NUM_BITS: usize = F::NUM_BITS * E;
    const GENERATOR: Self = Self(<F as Extended<E>>::GENERATOR);

    fn rand(mut rng: impl rand::RngCore) -> Self {
        std::array::from_fn(|_| F::rand(&mut rng)).into()
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|e| e.is_zero())
    }

    fn is_one(&self) -> bool {
        (*self - F::ONE).is_zero()
    }

    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let mut ret = Self::default();
        match E {
            2 => {
                let v0 = self[0].square() - self[1].square().mul_by_nonresidue();
                let t = v0.inverse().unwrap();
                ret[0] = self[0] * t;
                ret[1] = self[1] * -t;
            }
            3 => {
                let c0 = self[2].mul_by_nonresidue() * self[1].neg() + self[0].square();
                let c1 = self[2].square().mul_by_nonresidue() - (self[0] * self[1]);
                let c2 = self[1].square() - (self[0] * self[2]);
                let t = (self[2] * c1) + (self[1] * c2);
                let t = t.mul_by_nonresidue() + (self[0] * c0);
                let t = t.inverse().unwrap();
                ret[0] = self[0] * t;
                ret[1] = self[1] * t;
                ret[2] = self[2] * t;
            }
            _ => unimplemented!(),
        }
        Some(ret)
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut ret = Self::ZERO;
        let mut off = 0;
        for e in ret.iter_mut() {
            *e = F::from_bytes(&bytes[off..off + F::NUM_BYTES])?;
            off += F::NUM_BYTES;
        }
        Some(ret)
    }

    fn from_uniform_bytes(bytes: &[u8]) -> Self {
        bytes
            .chunks(F::NUM_BYTES * 2)
            .map(|chunk| F::from_uniform_bytes(chunk))
            .chain(std::iter::repeat(F::ZERO))
            .take(E)
            .collect()
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.iter().flat_map(|e| e.to_bytes()).collect()
    }
}

impl<const E: usize, F: Extended<E>> std::ops::Deref for Ext<E, F> {
    type Target = [F];
    fn deref(&self) -> &[F] {
        &self.0
    }
}

impl<const E: usize, F: Extended<E>> std::ops::DerefMut for Ext<E, F> {
    fn deref_mut(&mut self) -> &mut [F] {
        &mut self.0[..]
    }
}

impl<F: Field> ExtField<F> for F {
    fn as_slice(&self) -> &[F] {
        std::slice::from_ref(self)
    }
    fn set0(&mut self, e: F) {
        *self = e;
    }
}

impl<const E: usize, F: Extended<E>> Distribution<Ext<E, F>> for Standard
where
    Standard: Distribution<F>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Ext<E, F> {
        Ext(std::array::from_fn(|_| rng.gen()))
    }
}

impl<const E: usize, F: Extended<E>> std::ops::Neg for Ext<E, F> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<const E: usize, F: Extended<E>> std::ops::Neg for &Ext<E, F> {
    type Output = Ext<E, F>;
    fn neg(self) -> Self::Output {
        std::array::from_fn(|i| -self[i]).into()
    }
}

impl<const E: usize, F: Extended<E>> std::ops::Add<Ext<E, F>> for Ext<E, F> {
    type Output = Ext<E, F>;
    fn add(self, rhs: Ext<E, F>) -> Self::Output {
        std::array::from_fn(|i| self[i] + rhs[i]).into()
    }
}

impl<const E: usize, F: Extended<E>> std::ops::Add<F> for Ext<E, F> {
    type Output = Self;
    fn add(mut self, rhs: F) -> Self::Output {
        self[0] += rhs;
        self
    }
}

impl<const E: usize, F: Extended<E>> std::ops::AddAssign for Ext<E, F> {
    fn add_assign(&mut self, rhs: Self) {
        self.iter_mut().zip(rhs.iter()).for_each(|(l, &r)| *l += r);
    }
}

impl<const E: usize, F: Extended<E>> std::ops::AddAssign<F> for Ext<E, F> {
    fn add_assign(&mut self, rhs: F) {
        self[0] += rhs;
    }
}

impl<const E: usize, F: Extended<E>> std::ops::Sub for Ext<E, F> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        std::array::from_fn(|i| self[i] - rhs[i]).into()
    }
}

impl<const E: usize, F: Extended<E>> std::ops::Sub<F> for Ext<E, F> {
    type Output = Self;
    fn sub(mut self, rhs: F) -> Self::Output {
        self[0] -= rhs;
        self
    }
}

impl<const E: usize, F: Extended<E>> std::ops::Sub<F> for &Ext<E, F> {
    type Output = Ext<E, F>;
    fn sub(self, rhs: F) -> Self::Output {
        *self - rhs
    }
}

impl<const E: usize, F: Extended<E>> std::ops::SubAssign for Ext<E, F> {
    fn sub_assign(&mut self, rhs: Self) {
        self.iter_mut().zip(rhs.iter()).for_each(|(l, &r)| *l -= r);
    }
}

impl<const E: usize, F: Extended<E>> std::ops::SubAssign<F> for Ext<E, F> {
    fn sub_assign(&mut self, rhs: F) {
        self[0] -= rhs;
    }
}

impl<const E: usize, F: Extended<E>> std::ops::Mul<Ext<E, F>> for Ext<E, F> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut res = Ext::ZERO;
        match E {
            2 => {
                let v0 = self[0] * rhs[0];
                let v1 = self[1] * rhs[1];
                res[1] = (self[0] + self[1]) * (rhs[0] + rhs[1]) - (v0 + v1);
                res[0] = v0 + v1.mul_by_nonresidue();
            }
            3 => {
                let aa = self[0] * rhs[0];
                let bb = self[1] * rhs[1];
                let cc = self[2] * rhs[2];
                let t1 = (rhs[1] + rhs[2]) * (self[1] + self[2]) - (cc + bb);
                let t1 = aa + t1.mul_by_nonresidue();
                let t3 = (rhs[0] + rhs[2]) * (self[0] + self[2]) - (aa - bb + cc);
                let t2 = (rhs[0] + rhs[1]) * (self[0] + self[1]) - (aa + bb);
                let t2 = t2 + cc.mul_by_nonresidue();
                res[0] = t1;
                res[1] = t2;
                res[2] = t3;
            }
            _ => unimplemented!(),
        }
        res
    }
}

impl<const E: usize, F: Extended<E>> std::ops::MulAssign<Ext<E, F>> for Ext<E, F> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const E: usize, F: Extended<E>> std::ops::Mul<F> for Ext<E, F> {
    type Output = Self;
    fn mul(self, rhs: F) -> Self::Output {
        self.iter().map(|&l| l * rhs).collect()
    }
}

impl<const E: usize, F: Extended<E>> std::ops::Mul<&F> for Ext<E, F> {
    type Output = Self;
    fn mul(self, rhs: &F) -> Self::Output {
        self * *rhs
    }
}

impl<const E: usize, F: Extended<E>> std::ops::MulAssign<F> for Ext<E, F> {
    fn mul_assign(&mut self, rhs: F) {
        self.iter_mut().for_each(|l| *l *= rhs);
    }
}

impl<const E: usize, F: Extended<E>> core::iter::Sum for Ext<E, F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<'a, const E: usize, F: Extended<E>> core::iter::Sum<&'a Ext<E, F>> for Ext<E, F> {
    fn sum<I: Iterator<Item = &'a Ext<E, F>>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, &x| acc + x)
    }
}

impl<const E: usize, F: Extended<E>> core::iter::Product for Ext<E, F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<'a, const E: usize, F: Extended<E>> core::iter::Product<&'a Ext<E, F>> for Ext<E, F> {
    fn product<I: Iterator<Item = &'a Ext<E, F>>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, &x| acc * x)
    }
}
