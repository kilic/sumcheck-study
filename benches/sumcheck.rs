use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use mysnark::data::MatrixOwn;
use mysnark::field::Field;
use mysnark::sumcheck::algo1;
use mysnark::transcript::rust_crypto::RustCryptoWriter;
use mysnark::utils::n_rand;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rayon::iter::ParallelIterator;

fn seed_rng() -> impl rand::Rng {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    ChaCha20Rng::seed_from_u64(0)
}

fn setup<F: Field>(k: usize, d: usize) -> (MatrixOwn<F>, F)
where
    Standard: Distribution<F>,
{
    let n = 1 << k;
    let mut rng = seed_rng();
    let px = (0..d).map(|_| n_rand::<F>(&mut rng, n)).collect::<Vec<_>>();

    let mut mat = MatrixOwn::from_columns(&px);
    mat.reverse_bits();

    let sum = mat
        .par_iter()
        .map(|row| row.iter().product::<F>())
        .sum::<F>();

    (mat, sum)
}

fn sumcheck_prover_reversed(c: &mut Criterion) {
    type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
    type F = mysnark::field::goldilocks::Goldilocks;
    type EF = mysnark::field::goldilocks::Goldilocks2;

    let mut group = c.benchmark_group("sumcheck-group");
    let id = BenchmarkId::new("prove reversed", 25);
    group.bench_function(id, |b| {
        b.iter_batched(
            || setup::<F>(25, 3),
            |(mat, sum)| {
                let mut writer = Writer::init(b"");
                let (_red0, _rs): (EF, _) = algo1::reversed::prove(sum, &mat, &mut writer).unwrap();
            },
            BatchSize::LargeInput,
        );
    });
    let id = BenchmarkId::new("prove natural", 25);
    group.bench_function(id, |b| {
        b.iter_batched(
            || setup::<F>(25, 3),
            |(mat, sum)| {
                let mut writer = Writer::init(b"");
                let (_red0, _rs): (EF, _) = algo1::natural::prove(sum, mat, &mut writer).unwrap();
            },
            BatchSize::LargeInput,
        );
    });
}

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = sumcheck_prover_reversed
);
