use super::{Challenge, Reader, Writer};
use crate::field::Field;
use crate::Error;
use digest::{Digest, FixedOutputReset, Output};
use std::io::{Read, Write};

#[derive(Debug, Clone, Default)]
pub struct RustCrypto<D: Digest> {
    _0: std::marker::PhantomData<D>,
}

impl<D: Digest> RustCrypto<D> {
    pub fn new() -> Self {
        Self {
            _0: Default::default(),
        }
    }

    pub(crate) fn result_to_field<F: Field>(out: &Output<D>) -> F {
        F::from_uniform_bytes(out)
    }
}

impl<D: Digest + FixedOutputReset> RustCrypto<D> {
    fn cycle(h: &mut D) -> Output<D> {
        let ret = h.finalize_reset();
        Digest::update(h, &ret);
        ret
    }

    pub(crate) fn draw_field_element<F: Field>(h: &mut D) -> F {
        let ret = Self::cycle(h);
        RustCrypto::<D>::result_to_field(&ret)
    }
}

#[derive(Debug, Clone)]
pub struct RustCryptoWriter<W: Write, D: Digest + FixedOutputReset> {
    h: D,
    writer: W,
}

impl<W: Write + Default, D: Digest + FixedOutputReset> RustCryptoWriter<W, D> {
    pub fn init(prefix: impl AsRef<[u8]>) -> Self {
        RustCryptoWriter {
            h: D::new_with_prefix(prefix),
            writer: W::default(),
        }
    }
}

impl<W: Write, D: Digest + FixedOutputReset> RustCryptoWriter<W, D> {
    pub fn finalize(self) -> W {
        self.writer
    }

    fn update(&mut self, data: impl AsRef<[u8]>) {
        Digest::update(&mut self.h, data);
    }
}

impl<W: Write, D: Digest + FixedOutputReset, F: Field> Writer<F> for RustCryptoWriter<W, D> {
    fn no_contrib_write(&mut self, e: F) -> Result<(), Error> {
        let data = e.to_bytes();
        self.writer
            .write_all(data.as_ref())
            .map_err(|_| Error::Transcript)?;
        Ok(())
    }

    fn write(&mut self, e: F) -> Result<(), Error> {
        self.no_contrib_write(e)?;
        self.update(e.to_bytes());
        Ok(())
    }
}

impl<W: Write, D: Digest + FixedOutputReset, F: Field> Challenge<F> for RustCryptoWriter<W, D> {
    fn draw(&mut self) -> F {
        RustCrypto::draw_field_element(&mut self.h)
    }
}

#[derive(Debug, Clone)]
pub struct RustCryptoReader<R: Read, D: Digest + FixedOutputReset> {
    h: D,
    reader: R,
}

impl<R: Read, D: Digest + FixedOutputReset> RustCryptoReader<R, D> {
    pub fn init(reader: R, prefix: impl AsRef<[u8]>) -> Self {
        RustCryptoReader {
            h: D::new_with_prefix(prefix),
            reader,
        }
    }

    fn update(&mut self, data: impl AsRef<[u8]>) {
        Digest::update(&mut self.h, data);
    }
}

impl<R: Read, D: Digest + FixedOutputReset, F: Field> Challenge<F> for RustCryptoReader<R, D> {
    fn draw(&mut self) -> F {
        RustCrypto::draw_field_element(&mut self.h)
    }
}

impl<R: Read, D: Digest + FixedOutputReset, F: Field> Reader<F> for RustCryptoReader<R, D> {
    fn no_contrib_read(&mut self) -> Result<F, Error> {
        let mut data = vec![0u8; F::NUM_BYTES];
        self.reader
            .read_exact(data.as_mut())
            .map_err(|_| Error::Transcript)?;
        let e = F::from_bytes(&data).ok_or(Error::Transcript)?;
        Ok(e)
    }

    fn read(&mut self) -> Result<F, Error> {
        let e: F = self.no_contrib_read()?;
        self.update(e.to_bytes());
        Ok(e)
    }
}

#[test]
fn test_transcript() {
    use crate::field::goldilocks::Goldilocks;
    use rand::Rng;
    type F = Goldilocks;
    type Writer = RustCryptoWriter<Vec<u8>, sha2::Sha256>;
    type Reader<'a> = RustCryptoReader<&'a [u8], sha2::Sha256>;

    let a0: F = rand_core::OsRng.gen();
    let b0: F = rand_core::OsRng.gen();
    let c0: F = rand_core::OsRng.gen();
    let mut w = Writer::init("");

    w.write(a0).unwrap();
    w.write(b0).unwrap();
    let _: F = Challenge::<F>::draw(&mut w);
    w.write(c0).unwrap();
    let u0: F = Challenge::<F>::draw(&mut w);
    w.write(a0).unwrap();

    let stream = w.finalize();
    let mut r = Reader::init(&stream, "");
    let _: F = r.read().unwrap();
    let _: F = r.read().unwrap();
    let _: F = Challenge::<F>::draw(&mut r);
    let _: F = r.read().unwrap();
    let u1: F = Challenge::<F>::draw(&mut r);
    let a1: F = r.read().unwrap();

    assert_eq!(u0, u1);
    assert_eq!(a0, a1);
}

#[test]
fn test_keccak() {
    use crate::field::goldilocks::Goldilocks;
    use rand::Rng;
    type F = Goldilocks;
    type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
    type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;

    let a0: F = rand_core::OsRng.gen();
    let b0: F = rand_core::OsRng.gen();
    let c0: F = rand_core::OsRng.gen();
    let mut w = Writer::init("");

    w.write(a0).unwrap();
    w.write(b0).unwrap();
    let _: F = Challenge::<F>::draw(&mut w);
    w.write(c0).unwrap();
    let u0: F = Challenge::<F>::draw(&mut w);
    w.write(a0).unwrap();

    let stream = w.finalize();
    let mut r = Reader::init(&stream, "");
    let _: F = r.read().unwrap();
    let _: F = r.read().unwrap();
    let _: F = Challenge::<F>::draw(&mut r);
    let _: F = r.read().unwrap();
    let u1: F = Challenge::<F>::draw(&mut r);
    let a1: F = r.read().unwrap();

    assert_eq!(u0, u1);
    assert_eq!(a0, a1);
}
