use crate::{
    data::MatrixOwn,
    gate::{Gate, GenericGate, HandGateExample, HandGateMulExample, ZeroGate},
    transcript::rust_crypto::{RustCryptoReader, RustCryptoWriter},
    utils::n_rand,
};
use rayon::iter::ParallelIterator;
use tracing::info_span;

type F = crate::field::goldilocks::Goldilocks;
type EF = crate::field::goldilocks::Goldilocks2;

type Writer = RustCryptoWriter<Vec<u8>, sha3::Keccak256>;
type Reader<'a> = RustCryptoReader<&'a [u8], sha3::Keccak256>;

#[test]
fn test_sumcheck_prover1() {
    // crate::test::init_tracing();
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
        let mat = mat.clone();
        let gate = GenericGate::new(vec![vec![0usize.into(), 1usize.into(), 2usize.into()]], 3);
        let mut writer = Writer::init(b"");
        let (red0, rs) = super::algo1::prove(sum, mat, &gate, &mut writer).unwrap();
        let red1: EF = gate.eval(&rs, &mat0);
        assert_eq!(red0, red1);

        let proof = writer.finalize();
        let mut reader = Reader::init(&proof, b"");
        let (red2, _rs) = super::reduce_sumcheck_claim::<F, EF, _>(k, d, &mut reader).unwrap();
        assert_eq!(rs, _rs);
        assert_eq!(red0, red2)
    }

    {
        let mat = mat.clone();
        let gate = HandGateExample {};
        let mut writer = Writer::init(b"");
        let (red0, rs) = super::algo1::prove(sum, mat, &gate, &mut writer).unwrap();
        let red1: EF = gate.eval(&rs, &mat0);
        assert_eq!(red0, red1);

        let proof = writer.finalize();
        let mut reader = Reader::init(&proof, b"");
        let (red2, _rs) = super::reduce_sumcheck_claim::<F, EF, _>(k, d, &mut reader).unwrap();
        assert_eq!(rs, _rs);
        assert_eq!(red0, red2)
    }
}

#[test]
fn test_sumcheck_prover2() {
    // crate::test::init_tracing();
    let mut rng = crate::test::seed_rng();

    let k = 25;
    let d = 3;
    let n = 1 << k;
    let px = (0..d)
        .map(|_| n_rand::<EF>(&mut rng, n))
        .collect::<Vec<_>>();

    let mat = MatrixOwn::from_columns(&px);
    let mat0 = mat.clone();
    let sum = info_span!("sum").in_scope(|| {
        mat.par_iter()
            .map(|row| row.iter().product::<EF>())
            .sum::<EF>()
    });

    {
        let mat = mat.clone();
        let gate = GenericGate::new(vec![vec![0usize.into(), 1usize.into(), 2usize.into()]], 3);
        let mut writer = Writer::init(b"");
        let (red0, rs) = super::algo1::prove_nat(sum, mat, &gate, &mut writer).unwrap();
        let red1: EF = gate.eval(&rs, &mat0);
        assert_eq!(red0, red1);

        let proof = writer.finalize();
        let mut reader = Reader::init(&proof, b"");
        let (red2, _rs) = super::reduce_sumcheck_claim::<EF, EF, _>(k, d, &mut reader).unwrap();
        assert_eq!(rs, _rs);
        assert_eq!(red0, red2)
    }

    {
        let mat = mat.clone();
        let gate = HandGateExample {};
        let mut writer = Writer::init(b"");
        let (red0, rs) = super::algo1::prove_nat(sum, mat, &gate, &mut writer).unwrap();
        let red1: EF = gate.eval(&rs, &mat0);
        assert_eq!(red0, red1);

        let proof = writer.finalize();
        let mut reader = Reader::init(&proof, b"");
        let (red2, _rs) = super::reduce_sumcheck_claim::<EF, EF, _>(k, d, &mut reader).unwrap();
        assert_eq!(rs, _rs);
        assert_eq!(red0, red2)
    }
}

#[test]
fn test_zerocheck_prover() {
    // crate::test::init_tracing();
    let mut rng = crate::test::seed_rng();
    let k = 25;
    let n = 1 << k;
    let degree = 2;
    let ys: Vec<EF> = n_rand(&mut rng, k);

    let ax = n_rand::<F>(&mut rng, n);
    let bx = n_rand::<F>(&mut rng, n);
    let cx = ax
        .iter()
        .zip(bx.iter())
        .map(|(&a, &b)| a * b)
        .collect::<Vec<_>>();
    let mat = MatrixOwn::from_columns(&[ax, bx, cx]);

    let mut writer = Writer::init(b"");
    let gate = HandGateMulExample {};
    let (red0, rs) = super::zerocheck::prove(&mut writer, mat.clone(), gate, &ys).unwrap();
    let red1 = gate.eval(&rs, &mat);
    assert_eq!(red0, red1);

    let proof = writer.finalize();
    let mut reader = Reader::init(&proof, b"");
    let (red2, _rs) = super::reduce_zerocheck_claim::<F, EF, _>(&mut reader, degree, &ys).unwrap();
    assert_eq!(rs, _rs);
    assert_eq!(red0, red2)
}
