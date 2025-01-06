use crate::{
    data::MatrixOwn, field::Field, mle::MLE, transcript::rust_crypto::RustCryptoWriter,
    utils::n_rand, Natural,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use tracing::info_span;

type F = crate::field::goldilocks::Goldilocks;
type EF = crate::field::goldilocks::Goldilocks2;

type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;

pub fn eval_gate(mat: &MatrixOwn<F>, points: &[EF]) -> EF {
    assert_eq!(points.len(), mat.k());
    let eq = MLE::<_, Natural>::eq(points);

    let evals = eq
        .par_iter()
        .zip(mat.par_iter())
        .map(|(&r, row)| row.iter().map(|&e| r * e).collect::<Vec<_>>())
        .reduce(
            || vec![EF::ZERO; mat.width()],
            |acc, next| {
                acc.iter()
                    .zip(next.iter())
                    .map(|(&acc, &next)| acc + next)
                    .collect::<Vec<_>>()
            },
        );

    evals.iter().product()
}

#[test]
fn test_sumcheck() {
    crate::test::init_tracing();
    let mut rng = crate::test::seed_rng();

    let k = 25;
    let d = 3;
    let n = 1 << k;
    let px = (0..d).map(|_| n_rand::<F>(&mut rng, n)).collect::<Vec<_>>();

    let mat = MatrixOwn::from_columns(&px);
    let mat0 = mat.clone();

    let sum = info_span!("sum").in_scope(|| {
        mat.par_iter()
            .map(|row| row.iter().product::<F>())
            .sum::<F>()
    });

    {
        let mut mat = mat.clone();
        mat.reverse_bits();
        let mut writer = Writer::init(b"");
        let (red0, rs) = super::algo1::prove(sum, mat, &mut writer).unwrap();
        let red1 = eval_gate(&mat0, &rs);
        assert_eq!(red0, red1);
    }

    {
        let mat = mat.clone();
        let mut writer = Writer::init(b"");
        let (red0, rs) = super::algo1::prove2(sum, mat, &mut writer).unwrap();
        let red1 = eval_gate(&mat0, &rs);
        assert_eq!(red0, red1);
    }

    {
        let mut mat = mat.clone();
        mat.reverse_bits();
        let mut writer = Writer::init(b"");
        let (red0, rs) = super::algo1::prove3(sum, mat, &mut writer).unwrap();
        let red1 = eval_gate(&mat0, &rs);
        assert_eq!(red0, red1);
    }
}
