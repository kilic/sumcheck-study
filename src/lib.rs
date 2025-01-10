pub mod data;
pub mod field;
pub mod mle;
pub mod sumcheck;
pub mod transcript;
pub mod utils;
pub mod gate;

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Error {
    Transcript,
    Verify,
}

#[cfg(test)]
pub(crate) mod test {

    #[allow(dead_code)]
    pub(crate) fn seed_rng() -> impl rand::Rng {
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;
        ChaCha20Rng::seed_from_u64(0)
    }

    #[allow(dead_code)]
    pub(crate) fn os_rng() -> impl rand::Rng {
        rand_core::OsRng
    }

    #[allow(dead_code)]
    pub(crate) fn init_tracing() {
        use tracing_forest::util::LevelFilter;
        use tracing_forest::ForestLayer;
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        use tracing_subscriber::{EnvFilter, Registry};

        let env_filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();

        Registry::default()
            .with(env_filter)
            .with(ForestLayer::default())
            .init();
    }
}
