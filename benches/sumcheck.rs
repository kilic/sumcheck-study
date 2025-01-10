use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rayon::iter::ParallelIterator;
use sumcheck_study::data::MatrixOwn;
use sumcheck_study::field::Field;
use sumcheck_study::gate::{HandGateExample, HandGateMulExample};
use sumcheck_study::sumcheck::{algo1, zerocheck};
use sumcheck_study::transcript::rust_crypto::RustCryptoWriter;
use sumcheck_study::utils::n_rand;

fn seed_rng() -> impl rand::Rng {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    ChaCha20Rng::seed_from_u64(0)
}

fn sumcheck_prover(c: &mut Criterion) {
    type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
    type F = sumcheck_study::field::goldilocks::Goldilocks;
    type EF = sumcheck_study::field::goldilocks::Goldilocks2;
    let k = 25;

    fn setup<F: Field>(k: usize, d: usize) -> (MatrixOwn<F>, F)
    where
        Standard: Distribution<F>,
    {
        let n = 1 << k;
        let mut rng = seed_rng();
        let px = (0..d).map(|_| n_rand::<F>(&mut rng, n)).collect::<Vec<_>>();
        let mat = MatrixOwn::from_columns(&px);
        let sum = mat
            .par_iter()
            .map(|row| row.iter().product::<F>())
            .sum::<F>();

        (mat, sum)
    }

    let mut group = c.benchmark_group("sumcheck-group");
    let id = BenchmarkId::new("prove_sumcheck", k);
    group.bench_function(id, |b| {
        b.iter_batched(
            || setup::<F>(k, 3),
            |(mat, sum)| {
                let mut writer = Writer::init(b"");
                let gate = HandGateExample {};
                let (_red0, _rs): (EF, _) = algo1::prove(sum, mat, &gate, &mut writer).unwrap();
            },
            BatchSize::LargeInput,
        );
    });

    let id = BenchmarkId::new("prove_sumcheck_ext", k);
    group.bench_function(id, |b| {
        b.iter_batched(
            || setup::<EF>(25, 3),
            |(mat, sum)| {
                let mut writer = Writer::init(b"");
                let gate = HandGateExample {};
                let (_red0, _rs): (EF, _) = algo1::prove(sum, mat, &gate, &mut writer).unwrap();
            },
            BatchSize::LargeInput,
        );
    });
}

fn zerocheck_prover(c: &mut Criterion) {
    type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
    type F = sumcheck_study::field::goldilocks::Goldilocks;
    type EF = sumcheck_study::field::goldilocks::Goldilocks2;

    fn setup<F: Field>(k: usize) -> (MatrixOwn<F>, Vec<EF>)
    where
        Standard: Distribution<F>,
    {
        let mut rng = seed_rng();
        let n = 1 << k;
        let ys: Vec<EF> = n_rand::<EF>(&mut rng, k);
        let ax = n_rand::<F>(&mut rng, n);
        let bx = n_rand::<F>(&mut rng, n);
        let cx = ax
            .iter()
            .zip(bx.iter())
            .map(|(&a, &b)| a * b)
            .collect::<Vec<_>>();
        let mat = MatrixOwn::from_columns(&[ax, bx, cx]);
        (mat, ys)
    }

    let k = 25;
    let mut group = c.benchmark_group("sumcheck-group");
    let id = BenchmarkId::new("prove_zerocheck", k);
    group.bench_function(id, |b| {
        b.iter_batched(
            || setup::<F>(k),
            |(mat, ys)| {
                let mut writer = Writer::init(b"");
                let gate = HandGateMulExample {};
                let (_red0, _rs) = zerocheck::prove(&mut writer, mat.clone(), gate, &ys).unwrap();
            },
            BatchSize::LargeInput,
        );
    });
}

criterion_main!(benches);
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = sumcheck_prover, zerocheck_prover
);
