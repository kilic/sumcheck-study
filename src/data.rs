use crate::field::Field;
use crate::utils::{log2_strict, unsafe_allocate_zero_vec, BitReverse, TwoAdicSlice};
use core::fmt::Debug;
use rand::{distributions::Standard, prelude::Distribution};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

pub trait Storage<V> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<V> Storage<V> for &[V] {
    fn len(&self) -> usize {
        <[V]>::len(self)
    }
}

impl<V> Storage<V> for &mut [V] {
    fn len(&self) -> usize {
        <[V]>::len(self)
    }
}

impl<V> Storage<V> for Vec<V> {
    fn len(&self) -> usize {
        self.len()
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Matrix<V, S> {
    pub storage: S,
    width: usize,
    _marker: std::marker::PhantomData<V>,
}

pub type MatrixRef<'a, V> = Matrix<V, &'a [V]>;
pub type MatrixMut<'a, V> = Matrix<V, &'a mut [V]>;
pub type MatrixOwn<V> = Matrix<V, Vec<V>>;

impl<V> MatrixOwn<V> {
    pub fn rand(mut rng: impl rand::Rng, width: usize, height: usize) -> Self
    where
        Standard: Distribution<V>,
    {
        let values = (0..width * height).map(|_| rng.gen()).collect();
        Self::new(width, values)
    }

    pub fn zero(width: usize, k: usize) -> Self
    where
        V: Default,
    {
        let height = 1 << k;
        Self::new(width, unsafe_allocate_zero_vec(width * height))
    }

    pub fn as_mut(&mut self) -> MatrixMut<'_, V> {
        Matrix {
            storage: self.storage.as_mut(),
            width: self.width,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn resize(&mut self, k: usize)
    where
        V: Default + Clone,
    {
        self.storage.resize((1 << k) * self.width, V::default());
    }

    pub fn drop(&mut self, k: usize)
    where
        V: Default + Clone,
    {
        self.storage.truncate((1 << k) * self.width);
    }

    pub fn drop_hi(&mut self)
    where
        V: Default + Clone,
    {
        self.drop(self.k() - 1);
    }

    pub fn from_columns(px: &[Vec<V>]) -> MatrixOwn<V>
    where
        V: Field,
    {
        let width = px.len();
        let k = px.first().unwrap().k();
        let mut mat: MatrixOwn<V> = MatrixOwn::zero(width, k);
        mat.par_iter_mut().enumerate().for_each(|(i, row)| {
            row.iter_mut()
                .zip(px.iter())
                .for_each(|(e0, pix)| *e0 = pix[i]);
        });
        mat
    }
}

impl<V, S: Storage<V>> Matrix<V, S> {
    pub fn new(width: usize, storage: S) -> Self {
        Self {
            storage,
            width,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn owned(&self) -> MatrixOwn<V>
    where
        S: AsRef<[V]>,
        V: Clone,
    {
        Matrix::new(self.width, self.storage.as_ref().to_vec())
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        assert!(self.storage.len() % self.width == 0);
        self.storage.len() / self.width
    }

    pub fn k(&self) -> usize {
        log2_strict(self.height())
    }

    pub fn row(&self, i: usize) -> &[V]
    where
        S: AsRef<[V]>,
    {
        let start = i * self.width;
        let end = start + self.width;
        &self.storage.as_ref()[start..end]
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [V]
    where
        S: AsMut<[V]>,
    {
        let start = i * self.width;
        let end = start + self.width;
        &mut self.storage.as_mut()[start..end]
    }

    pub fn chunks(&self, size: usize) -> impl Iterator<Item = MatrixRef<V>>
    where
        S: AsRef<[V]>,
    {
        self.storage
            .as_ref()
            .chunks(size * self.width)
            .map(|inner| Matrix::new(self.width, inner))
    }

    pub fn chunks_mut(&mut self, size: usize) -> impl Iterator<Item = MatrixMut<V>>
    where
        S: AsMut<[V]>,
    {
        self.storage
            .as_mut()
            .chunks_mut(size * self.width)
            .map(|inner| Matrix::new(self.width, inner))
    }

    pub fn chunk_pair_mut(&mut self) -> impl IndexedParallelIterator<Item = (&mut [V], &mut [V])>
    where
        S: AsMut<[V]>,
        V: Send + Sync,
    {
        self.storage
            .as_mut()
            .par_chunks_mut(2 * self.width)
            .map(|inner| inner.split_at_mut(self.width))
    }

    pub fn chunk_pair(&self) -> impl IndexedParallelIterator<Item = (&[V], &[V])>
    where
        S: AsRef<[V]> + Send + Sync,
        V: Send + Sync,
    {
        self.storage
            .as_ref()
            .par_chunks(2 * self.width)
            .map(|inner| inner.split_at(self.width))
    }

    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = &[V]>
    where
        S: AsRef<[V]>,
        V: Send + Sync,
    {
        self.storage.as_ref().par_chunks(self.width)
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &[V]>
    where
        S: AsRef<[V]>,
    {
        self.storage.as_ref().chunks(self.width)
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut [V]>
    where
        S: AsMut<[V]>,
    {
        self.storage.as_mut().chunks_mut(self.width)
    }

    pub fn par_iter_mut(&mut self) -> impl IndexedParallelIterator<Item = &mut [V]>
    where
        S: AsMut<[V]>,
        V: Send + Sync,
    {
        self.storage.as_mut().par_chunks_mut(self.width)
    }

    pub fn par_chunks_mut(
        &mut self,
        size: usize,
    ) -> impl IndexedParallelIterator<Item = MatrixMut<V>>
    where
        S: AsMut<[V]>,
        V: Send + Sync,
    {
        self.storage
            .as_mut()
            .par_chunks_mut(self.width * size)
            .map(|chunk| MatrixMut::new(self.width, chunk))
    }

    pub fn par_chunks(&self, size: usize) -> impl IndexedParallelIterator<Item = MatrixRef<V>>
    where
        S: AsRef<[V]> + Send + Sync,
        V: Send + Sync,
    {
        self.storage
            .as_ref()
            .par_chunks(self.width * size)
            .map(|chunk| MatrixRef::new(self.width, chunk))
    }

    pub fn split_mut(&mut self, k: usize) -> (MatrixMut<V>, MatrixMut<V>)
    where
        S: AsMut<[V]>,
    {
        let width = self.width();
        let (v0, v1) = self.storage.as_mut().split_at_mut((1 << k) * width);
        (Matrix::new(width, v0), Matrix::new(width, v1))
    }

    pub fn split_mut_half(&mut self) -> (MatrixMut<V>, MatrixMut<V>)
    where
        S: AsMut<[V]>,
    {
        self.split_mut(self.k() - 1)
    }

    pub fn split(&self, k: usize) -> (MatrixRef<V>, MatrixRef<V>)
    where
        S: AsRef<[V]>,
    {
        let width = self.width();
        let (v0, v1) = self.storage.as_ref().split_at((1 << k) * width);
        (Matrix::new(width, v0), Matrix::new(width, v1))
    }

    pub fn split_half(&self) -> (MatrixRef<V>, MatrixRef<V>)
    where
        S: AsRef<[V]>,
    {
        self.split(self.k() - 1)
    }

    pub fn reverse_bits(&mut self)
    where
        S: AsMut<[V]>,
        V: Clone + Send + Sync,
    {
        self.storage.as_mut().reverse_bits_2d(self.width);
    }
}
