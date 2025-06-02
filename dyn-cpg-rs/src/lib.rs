pub mod cpg;
pub mod languages;
pub mod resource;

pub mod logging {
    use tracing_subscriber::EnvFilter;

    pub fn init() {
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .init();
    }
}
