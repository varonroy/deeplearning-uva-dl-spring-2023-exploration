use burn::module::ConstantRecord;
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::ReLU;
use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Int, Tensor},
    train::ClassificationOutput,
};

pub mod inception;
pub mod res;

pub mod cbuilder {
    use burn::nn::conv::Conv2dConfig;

    pub fn conv_2d(
        c_in: usize,
        c_out: usize,
        kernel_size: usize,
        padding: usize,
        stride: usize,
        bias: bool,
    ) -> Conv2dConfig {
        Conv2dConfig::new([c_in, c_out], [kernel_size, kernel_size])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(padding, padding))
            .with_stride([stride, stride])
            .with_bias(bias)
    }

    #[macro_export]
    macro_rules! conv2d {
        (
            c_in=$in_c:expr,
            c_out=$out_c:expr,
            kernel_size=$kernel_size:expr,
            padding=$padding:expr,
            stride=$stride:expr,
            bias=$bias:expr,
        ) => {
            // TODO: usize above function
            // conv_2d(&c_in, &c_out, &kernel_size, )
            Conv2dConfig::new([$in_c, $out_c], [$kernel_size, $kernel_size])
                .with_padding(burn::nn::PaddingConfig2d::Explicit($padding, $padding))
                .with_stride([$stride, $stride])
        };
    }
}

pub trait ForwardModule<B: Backend, const NI: usize, const NO: usize>: Module<B> {
    fn forward(&self, x: Tensor<B, NI>) -> Tensor<B, NO>;
}

pub trait InitWith {
    type Module;
    type Record;

    fn init_with(&self, record: Self::Record) -> Self::Module;
}

#[derive(Debug, Module, Clone, Copy)]
pub struct Flatten<const DI: usize, const DO: usize> {
    pub start_dim: usize,
    pub end_dim: usize,
}

impl<const DI: usize, const DO: usize> Flatten<DI, DO> {
    pub fn forward<B: Backend>(&self, x: Tensor<B, DI>) -> Tensor<B, DO> {
        x.flatten(self.start_dim, self.end_dim)
    }
}

impl InitWith for AdaptiveAvgPool2dConfig {
    type Module = AdaptiveAvgPool2d;
    type Record = ConstantRecord;

    fn init_with(&self, _record: ConstantRecord) -> AdaptiveAvgPool2d {
        self.init()
    }
}

#[derive(Debug, Config)]
pub struct FlattenConfig42 {
    pub start_dim: usize,
    pub end_dim: usize,
}

impl FlattenConfig42 {
    pub fn init(&self) -> Flatten<4, 2> {
        Flatten {
            start_dim: self.start_dim,
            end_dim: self.end_dim,
        }
    }

    pub fn init_with(&self, _record: ConstantRecord) -> Flatten<4, 2> {
        Flatten {
            start_dim: self.start_dim,
            end_dim: self.end_dim,
        }
    }
}

#[derive(Debug, Config, Copy)]
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

    pub fn init_with(self, _record: ConstantRecord) -> Activation {
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

macro_rules! impl_activation_config {
    ($config_name:ident, $activation:ident) => {
        #[derive(Debug, Config)]
        pub struct $config_name {}

        impl $config_name {
            pub fn init(&self) -> $activation {
                $activation::new()
            }

            pub fn init_with(&self, _record: burn::module::ConstantRecord) -> $activation {
                $activation::new()
            }
        }
    };
}

impl_activation_config!(ReLUConfig, ReLU);

pub trait ClassificationModel<B: Backend> {
    fn forward_classification(
        &self,
        img: Tensor<B, 4>,
        label: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B>;
}

#[macro_export]
macro_rules! init_config_layer {
    ($layer_name:ident, $layer_module:ty, $layer_config:ty) => {
        $layer_name: self.$layer_name.init()
    }
}

#[macro_export]
macro_rules! impl_config {
    ($name:ident,
         $config_name:ident,
         $record_name:ident,
         $input_dim:literal => $output_dim:literal,
         $(($layer_name:ident, $layer_module:ty, $layer_config:ty)),*) => {

        // #[derive(Debug, Config)]
        #[derive(Config)]
        pub struct $config_name {
            $(pub $layer_name: $layer_config,)*
        }

        impl $config_name {
            pub fn init<B: Backend>(&self) -> $name<B> {
                $name {
                    $($layer_name: self.$layer_name.init(),)*
                }
            }

            pub fn init_with<B: Backend>(&self, record: $record_name<B>) -> $name<B> {
                $name {
                    $($layer_name: self.$layer_name.init_with(record.$layer_name),)*
                }
            }
        }
    };
}

/// creates a sequential module.
/// ```rust
///     sequential!(
///         MyBlock,
///         MyBlockConfig,
///         MyBlockRecord,
///         4 => 4,
///         (l_1, burn::nn::Linear<B>, burn::nn::LinearConfig),
///         (act_1, burn::nn::ReLU, ReLUConfig),
///         (bn_1, burn::nn::BatchNorm<B, 2>, burn::nn::BatchNormConfig),
///         (l_2, burn::nn::Linear<B>, burn::nn::LinearConfig)
///     );
/// ```
#[macro_export]
macro_rules! sequential {
    ($name:ident,
         $config_name:ident,
         $record_name:ident,
         $input_dim:literal => $output_dim:literal,
         $(($layer_name:ident, $layer_module:ty, $layer_config:ty)),*) => {

        crate::impl_config!(
            $name,
            $config_name,
            $record_name,
            $input_dim => $output_dim,
            $(($layer_name, $layer_module, $layer_config)),*
        );


        #[derive(Debug, Module)]
        pub struct $name<B: Backend> {
            $(pub $layer_name: $layer_module,)*
        }

        impl<B: Backend> $name<B> {
            pub fn forward(&self, x: Tensor<B,$input_dim >) -> Tensor<B, $output_dim> {
                let x = x;
                $(let x = self.$layer_name.forward(x);)*
                x
            }
        }
    };
}

#[macro_export]
macro_rules! sequential_no_forward {
    ($name:ident,
         $config_name:ident,
         $record_name:ident,
         $input_dim:literal => $output_dim:literal,
         $(($layer_name:ident, $layer_module:ty, $layer_config:ty)),*) => {

        crate::impl_config!(
            $name,
            $config_name,
            $record_name,
            $input_dim => $output_dim,
            $(($layer_name, $layer_module, $layer_config)),*
        );


        #[derive(Debug, Module)]
        pub struct $name<B: Backend> {
            $(pub $layer_name: $layer_module,)*
        }

        impl<B: Backend> $name<B> {
            pub fn forward(&self, x: Tensor<B,$input_dim >) -> Tensor<B, $output_dim> {
                let x = x;
                $(let x = self.$layer_name.forward(x);)*
                x
            }
        }
    };
}

/// creates a sequential module.
/// ```rust
///     res!(
///         MyBlock,
///         MyBlockConfig,
///         MyBlockRecord,
///         4 => 4,
///         (l_1, burn::nn::Linear<B>, burn::nn::LinearConfig),
///         (act_1, burn::nn::ReLU, ReLUConfig),
///         (bn_1, burn::nn::BatchNorm<B, 2>, burn::nn::BatchNormConfig),
///         (l_2, burn::nn::Linear<B>, burn::nn::LinearConfig)
///     );
/// ```
#[macro_export]
macro_rules! res {
    ($name:ident,
         $config_name:ident,
         $record_name:ident,
         $input_dim:literal => $output_dim:literal,
         $(($layer_name:ident, $layer_module:ty, $layer_config:ty)),*) => {

        crate::impl_config!(
            $name,
            $config_name,
            $record_name,
            $input_dim => $output_dim,
            $(($layer_name, $layer_module, $layer_config)),*
        );

        #[derive(Debug, Module)]
        pub struct $name<B: Backend> {
            $(pub $layer_name: $layer_module,)*
        }
    };
}
