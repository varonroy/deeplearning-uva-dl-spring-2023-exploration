use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Int, Tensor},
    train::ClassificationOutput,
};

pub mod inception;

#[derive(Config, Copy)]
pub enum ActivationConfig {
    ReLU,
    Sigmoid,
    Tanh,
    Gelu,
}

impl ActivationConfig {
    pub fn init(self) -> Activation {
        match self {
            Self::ReLU => Activation::ReLU,
            Self::Sigmoid => Activation::Sigmoid,
            Self::Tanh => Activation::Tanh,
            Self::Gelu => Activation::Gelu,
        }
    }
}

#[derive(Debug, Module, Clone)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Gelu,
}

impl Activation {
    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Self::ReLU => burn::tensor::activation::relu(x),
            Self::Sigmoid => burn::tensor::activation::sigmoid(x),
            Self::Tanh => burn::tensor::activation::tanh(x),
            Self::Gelu => burn::tensor::activation::gelu(x),
        }
    }
}

pub trait ClassificationModel<B: Backend> {
    fn forward_classification(
        &self,
        img: Tensor<B, 4>,
        label: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B>;
}
