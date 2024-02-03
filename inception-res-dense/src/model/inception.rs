use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        loss::CrossEntropyLossConfig,
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d,
    },
    tensor::{backend::Backend, Int, Tensor},
    train::ClassificationOutput,
};
use dl_utils::nn::{Activation, ActivationConfig, Flatten, Flatten42Config};
use dl_utils_burn_sequential::SequentialForward;

use crate::data::cifar10::NUM_CLASSES;

use super::ClassificationModel;

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 4)]
pub struct ConvBlock<B: Backend> {
    pub conv: Conv2d<B>,
    pub bn: BatchNorm<B, 2>,
    pub act: Activation,
}

impl ConvBlockConfig {
    pub fn new2(
        c_in: usize,
        c_out: usize,
        kernel_size: usize,
        padding: usize,
        act: ActivationConfig,
    ) -> Self {
        Self {
            conv: Conv2dConfig::new([c_in, c_out], [kernel_size, kernel_size])
                .with_padding(PaddingConfig2d::Explicit(padding, padding)),
            bn: BatchNormConfig::new(c_out),
            act,
        }
    }
}

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 4)]
pub struct ConvNBlock<B: Backend> {
    conv_a: ConvBlock<B>,
    conv_b: ConvBlock<B>,
}

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 4)]
pub struct ConvPoolBlock<B: Backend> {
    pool: MaxPool2d,
    conv: ConvBlock<B>,
}

impl InceptionBlockConfig {
    pub fn new2(
        in_c: usize,
        conv_3_reduction: usize,
        conv_5_reduction: usize,
        conv_1_out: usize,
        conv_3_out: usize,
        conv_5_out: usize,
        conv_pool_out: usize,
        act: ActivationConfig,
    ) -> Self {
        Self {
            conv_1: ConvBlockConfig::new2(in_c, conv_1_out, 1, 0, act),
            conv_3: ConvNBlockConfig::new(
                ConvBlockConfig::new2(in_c, conv_3_reduction, 1, 0, act),
                ConvBlockConfig::new2(conv_3_reduction, conv_3_out, 3, 1, act),
            ),
            conv_5: ConvNBlockConfig::new(
                ConvBlockConfig::new2(in_c, conv_5_reduction, 1, 0, act),
                ConvBlockConfig::new2(conv_5_reduction, conv_5_out, 5, 2, act),
            ),
            conv_pool: ConvPoolBlockConfig::new(
                MaxPool2dConfig::new([3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .with_strides([1, 1]),
                ConvBlockConfig::new2(in_c, conv_pool_out, 1, 0, act),
            ),
        }
    }
}

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 4)]
#[manual_forward]
pub struct InceptionBlock<B: Backend> {
    conv_1: ConvBlock<B>,
    conv_3: ConvNBlock<B>,
    conv_5: ConvNBlock<B>,
    conv_pool: ConvPoolBlock<B>,
}

impl<B: Backend> InceptionBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        Tensor::cat(
            vec![
                self.conv_1.forward(x.clone()),
                self.conv_3.forward(x.clone()),
                self.conv_5.forward(x.clone()),
                self.conv_pool.forward(x.clone()),
            ],
            1,
        )
    }
}

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 4)]
pub struct ModelInceptionModule<B: Backend> {
    inception_1: Vec<InceptionBlock<B>>,
    pool_1: MaxPool2d,
    inception_2: Vec<InceptionBlock<B>>,
    pool_2: MaxPool2d,
    inception_3: Vec<InceptionBlock<B>>,
}

type Flatten42 = Flatten<4, 2>;

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 2)]
pub struct ModelOutputModule<B: Backend> {
    pool: AdaptiveAvgPool2d,
    flatten: Flatten42,
    linear: Linear<B>,
}

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 2)]
pub struct Model<B: Backend> {
    input: ConvBlock<B>,
    inception: ModelInceptionModule<B>,
    output: ModelOutputModule<B>,
}

impl<B: Backend> ClassificationModel<B> for Model<B> {
    fn forward_classification(
        &self,
        x: Tensor<B, 4>,
        label: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let out = self.forward(x);
        let loss = CrossEntropyLossConfig::new()
            .init(&out.device())
            .forward(out.clone(), label.clone());
        ClassificationOutput::new(loss, out, label)
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        let act = ActivationConfig::ReLU;
        let pool = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1));

        ModelConfig::new(
            ConvBlockConfig::new2(3, 64, 3, 1, act),
            ModelInceptionModuleConfig::new(
                vec![
                    InceptionBlockConfig::new2(64, 32, 16, 16, 32, 8, 8, act),
                    InceptionBlockConfig::new2(64, 32, 16, 24, 48, 12, 12, act),
                ],
                pool.clone(),
                vec![
                    InceptionBlockConfig::new2(96, 32, 16, 24, 48, 12, 12, act),
                    InceptionBlockConfig::new2(96, 32, 16, 16, 48, 16, 16, act),
                    InceptionBlockConfig::new2(96, 32, 16, 16, 48, 16, 16, act),
                    InceptionBlockConfig::new2(96, 32, 16, 32, 48, 24, 24, act),
                ],
                pool.clone(),
                vec![
                    InceptionBlockConfig::new2(128, 48, 16, 32, 64, 16, 16, act),
                    InceptionBlockConfig::new2(128, 48, 16, 32, 64, 16, 16, act),
                ],
            ),
            ModelOutputModuleConfig::new(
                AdaptiveAvgPool2dConfig::new([1, 1]),
                Flatten42Config::new(1, 3),
                LinearConfig::new(128, NUM_CLASSES as _),
            ),
        )
    }
}
