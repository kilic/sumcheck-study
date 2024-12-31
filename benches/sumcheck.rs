use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use mysnark::data::MatrixOwn;
use mysnark::sumcheck::algo1;
use mysnark::transcript::rust_crypto::RustCryptoWriter;
use mysnark::utils::n_rand;
use rayon::iter::ParallelIterator;

fn seed_rng() -> impl rand::Rng {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    ChaCha20Rng::seed_from_u64(0)
}

fn sumcheck_prover(c: &mut Criterion) {
    type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
    type F = mysnark::field::goldilocks::Goldilocks;

    let mut group = c.benchmark_group("group-xxx");
    let id = BenchmarkId::new("some-function-name", 25);
    group.bench_function(id, |b| {
        b.iter_batched(
            || {
                let mut rng = seed_rng();
                let k = 25;
                let d = 3;
                let n = 1 << k;
                let px = (0..d).map(|_| n_rand::<F>(&mut rng, n)).collect::<Vec<_>>();

                let mut mat = MatrixOwn::from_columns(&px);
                mat.reverse_bits();

                let sum = mat
                    .par_iter()
                    .map(|row| row.iter().product::<F>())
                    .sum::<F>();

                (mat, sum)
            },
            |(mat, sum)| {
                let mut writer = Writer::init(b"");
                let (_red0, _rs) = algo1::prove(sum, mat, &mut writer).unwrap();
            },
            BatchSize::LargeInput,
        );
    });
}

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = sumcheck_prover
);
