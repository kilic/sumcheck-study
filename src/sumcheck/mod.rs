use crate::{
    arithmetic::interpolate,
    field::{ExtField, Field},
};

pub mod algo1;

pub(super) fn extrapolate<F: Field, EF: ExtField<F>>(evals: &[F], target: EF) -> EF {
    let points = (0..evals.len())
        .map(|i| F::from(i as u64))
        .collect::<Vec<_>>();
    let sx = interpolate(&points, evals);
    target.horner(&sx)
}

#[cfg(test)]
mod test {

    use crate::{
        data::MatrixOwn,
        field::{Ext, Field},
        mle::MLE,
        transcript::rust_crypto::RustCryptoWriter,
        utils::n_rand,
        Natural,
    };
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
    use tracing::info_span;

    type F = crate::field::goldilocks::Goldilocks;
    type EF = Ext<2, F>;
    type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;

    pub fn eval_gate(mat: &MatrixOwn<F>, points: &[EF]) -> EF {
        assert_eq!(points.len(), mat.k());
        let eq = MLE::<_, Natural>::eq(points);

        let evals = eq
            .par_iter()
            .zip(mat.par_iter())
            .map(|(&r, row)| row.iter().map(|e| r * e).collect::<Vec<_>>())
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

        let k = 22;
        let d = 3;
        let n = 1 << k;
        let px = (0..d).map(|_| n_rand::<F>(&mut rng, n)).collect::<Vec<_>>();

        let mut mat = MatrixOwn::from_columns(&px);
        let mat0 = mat.clone();

        mat.reverse_bits();
        let sum = info_span!("sum").in_scope(|| {
            mat.par_iter()
                .map(|row| row.iter().product::<F>())
                .sum::<F>()
        });

        let mut writer = Writer::init(b"");

        let (red0, rs) = super::algo1::prove(sum, mat, &mut writer).unwrap();
        let red1 = eval_gate(&mat0, &rs);
        assert_eq!(red0, red1);
    }
}
