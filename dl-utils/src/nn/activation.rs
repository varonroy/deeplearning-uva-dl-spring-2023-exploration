use burn::{
    config::Config,
    module::{ConstantRecord, Module},
    nn::ReLU,
    tensor::{backend::Backend, Tensor},
};
use derive_new::new;

macro_rules! impl_activation {
    ($name:ident, $fn:expr) => {
        #[derive(Debug, Module, Clone, Copy, new)]
        pub struct $name;

        impl $name {
            pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
                $fn(x)
            }
        }
    };
}

impl_activation!(Sigmoid, burn::tensor::activation::sigmoid);
impl_activation!(Tanh, burn::tensor::activation::tanh);
impl_activation!(Gelu, burn::tensor::activation::gelu);

macro_rules! impl_activation_config {
    ($config_name:ident, $activation:ident) => {
        #[derive(Debug, Config)]
        pub struct $config_name {}

        impl $config_name {
            pub fn init(&self) -> $activation {
                $activation::new()
            }

            pub fn init_with(&self, _record: ConstantRecord) -> $activation {
                $activation::new()
            }
        }
    };
}

impl_activation_config!(ReLUConfig, ReLU);
impl_activation_config!(SigmoidConfig, Sigmoid);
impl_activation_config!(TanhConfig, Tanh);
impl_activation_config!(GeluConfig, Gelu);

#[derive(Debug, Config, Copy)]
pub enum ActivationConfig {
    ReLU,
    Sigmoid,
    Tanh,
    Gelu,
}

impl ActivationConfig {
    pub fn init(&self) -> Activation {
        match self {
            Self::ReLU => Activation::ReLU,
            Self::Sigmoid => Activation::Sigmoid,
            Self::Tanh => Activation::Tanh,
            Self::Gelu => Activation::Gelu,
        }
    }
}

#[derive(Debug, Module, Clone, Copy)]
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
