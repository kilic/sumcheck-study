pub mod rust_crypto;

pub trait Challenge<F> {
    fn draw(&mut self) -> F;
    fn draw_n(&mut self, n: usize) -> Vec<F> {
        (0..n).map(|_| self.draw()).collect()
    }
}

pub trait Writer<T> {
    fn write(&mut self, el: T) -> Result<(), crate::Error>;
    fn no_contrib_write(&mut self, el: T) -> Result<(), crate::Error>;
    fn write_n(&mut self, el: &[T]) -> Result<(), crate::Error>
    where
        T: Copy,
    {
        el.iter().try_for_each(|&e| self.write(e))
    }
    fn no_contrib_write_n(&mut self, el: &[T]) -> Result<(), crate::Error>
    where
        T: Copy,
    {
        el.iter().try_for_each(|&e| self.no_contrib_write(e))
    }
}

pub trait Reader<T> {
    fn read(&mut self) -> Result<T, crate::Error>;
    fn no_contrib_read(&mut self) -> Result<T, crate::Error>;
    fn read_n(&mut self, n: usize) -> Result<Vec<T>, crate::Error> {
        (0..n).map(|_| self.read()).collect()
    }
    fn no_contrib_read_n(&mut self, n: usize) -> Result<Vec<T>, crate::Error> {
        (0..n).map(|_| self.no_contrib_read()).collect()
    }
}
