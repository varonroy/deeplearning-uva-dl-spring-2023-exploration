pub mod activation;
pub mod flatten;

pub use activation::*;
pub use flatten::*;

use burn::tensor::{backend::Backend, Tensor};

#[derive(
    Debug, Clone, Copy, derive_new::new, serde::Deserialize, serde::Serialize, burn::module::Module,
)]
pub struct Identity {}

impl Identity {
    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        x
    }
}
