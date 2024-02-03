use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        loss::CrossEntropyLossConfig,
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, AvgPool2d, AvgPool2dConfig},
        BatchNorm, BatchNormConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, Int, Tensor},
    train::ClassificationOutput,
};
use dl_utils::pipe;
use dl_utils::{
    conv_2d,
    nn::{Activation, ActivationConfig, Flatten, Flatten42Config},
};
use dl_utils_burn_sequential::SequentialForward;

use crate::data::cifar10::NUM_CLASSES;

use super::ClassificationModel;

#[derive(Debug, Module, SequentialForward)]
#[manual_forward]
#[dims(4, 4)]
pub struct DenseLayer<B: Backend> {
    bn_1: BatchNorm<B, 2>,
    act_1: Activation,
    conv_1: Conv2d<B>,
    bn_2: BatchNorm<B, 2>,
    act_2: Activation,
    conv_2: Conv2d<B>,
}

impl<B: Backend> DenseLayer<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let z = x.clone();
        let z = pipe!(
            z,
            self.bn_1,
            self.act_1,
            self.conv_1,
            self.bn_2,
            self.act_2,
            self.conv_2,
        );
        Tensor::cat(vec![z, x], 1)
    }
}

impl DenseLayerConfig {
    pub fn new2(c_in: usize, bn_size: usize, growth_rate: usize, act: ActivationConfig) -> Self {
        Self {
            bn_1: BatchNormConfig::new(c_in),
            act_1: act,
            conv_1: conv_2d!(c_in, bn_size * growth_rate, kernel_size = 1, bias = false),
            bn_2: BatchNormConfig::new(bn_size * growth_rate),
            act_2: act,
            conv_2: conv_2d!(
                bn_size * growth_rate,
                growth_rate,
                kernel_size = 3,
                padding = 1,
                bias = false
            ),
        }
    }
}

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 4)]
pub struct DenseBlock<B: Backend> {
    layers: Vec<DenseLayer<B>>,
}

impl DenseBlockConfig {
    pub fn new2(
        c_in: usize,
        num_layers: usize,
        bn_size: usize,
        growth_rate: usize,
        act: ActivationConfig,
    ) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|layer_idx| {
                    DenseLayerConfig::new2(
                        c_in + layer_idx * growth_rate,
                        bn_size,
                        growth_rate,
                        act,
                    )
                })
                .collect(),
        }
    }
}

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 4)]
pub struct TransitionLayer<B: Backend> {
    bn: BatchNorm<B, 2>,
    act: Activation,
    conv2d: Conv2d<B>,
    pool: AvgPool2d,
}

impl TransitionLayerConfig {
    pub fn new2(c_in: usize, c_out: usize, act: ActivationConfig) -> Self {
        Self::new(
            BatchNormConfig::new(c_in),
            act,
            conv_2d!(c_in, c_out, kernel_size = 1, bias = false),
            AvgPool2dConfig::new([2, 2]).with_strides([2, 2]),
        )
    }
}

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 2)]
struct ModelOutput<B: Backend> {
    bn: BatchNorm<B, 2>,
    act: Activation,
    pool: AdaptiveAvgPool2d,
    flatten: Flatten42,
    linear: Linear<B>,
}

type Flatten42 = Flatten<4, 2>;

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 2)]
pub struct Model<B: Backend> {
    input: Conv2d<B>,
    body: Vec<(DenseBlock<B>, Option<TransitionLayer<B>>)>,
    output: ModelOutput<B>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        let num_layers = [6, 6, 6, 6];
        let bn_size = 2;
        let growth_rate = 16;
        let act = ActivationConfig::ReLU;

        let mut c_hidden = growth_rate * bn_size;

        Self {
            input: conv_2d!(3, c_hidden, kernel_size = 3, padding = 1),
            body: num_layers
                .into_iter()
                .enumerate()
                .map(|(block_idx, layer_num_layers)| {
                    let dense = DenseBlockConfig::new2(
                        c_hidden,
                        layer_num_layers,
                        bn_size,
                        growth_rate,
                        act,
                    );
                    c_hidden = c_hidden + layer_num_layers * growth_rate;
                    let transition = if block_idx < num_layers.len() {
                        let transition = TransitionLayerConfig::new2(c_hidden, c_hidden / 2, act);
                        c_hidden = c_hidden / 2;
                        Some(transition)
                    } else {
                        None
                    };
                    (dense, transition)
                })
                .collect(),
            output: ModelOutputConfig::new(
                BatchNormConfig::new(c_hidden),
                act,
                AdaptiveAvgPool2dConfig::new([1, 1]),
                Flatten42Config::new(1, 3),
                LinearConfig::new(c_hidden, NUM_CLASSES as _),
            ),
        }
    }
}

impl<B: Backend> ClassificationModel<B> for Model<B> {
    fn forward_classification(
        &self,
        x: Tensor<B, 4>,
        label: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(x);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), label.clone());

        ClassificationOutput::new(loss, output, label)
    }
}
