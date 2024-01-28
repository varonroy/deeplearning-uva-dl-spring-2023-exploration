use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        loss::CrossEntropyLoss,
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d,
    },
    tensor::{backend::Backend, Int, Tensor},
    train::ClassificationOutput,
};
use itertools::izip;

use crate::data::cifar10::NUM_CLASSES;

use super::ClassificationModel;
use super::{Activation, ActivationConfig};

#[derive(Debug, Module)]
pub struct ConvBlock<B: Backend> {
    pub conv: Conv2d<B>,
    pub bn: BatchNorm<B, 2>,
    pub act: Activation,
}

impl<B: Backend> ConvBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        let x = self.act.forward(x);
        x
    }
}

#[derive(Config, Copy)]
pub struct ConvBlockConfig {
    pub c_in: usize,
    pub c_out: usize,
    pub kernel_size: usize,
    pub padding: usize,
    pub act: ActivationConfig,
}

impl ConvBlockConfig {
    pub fn init<B: Backend>(self) -> ConvBlock<B> {
        ConvBlock {
            conv: Conv2dConfig::new(
                [self.c_in, self.c_out],
                [self.kernel_size, self.kernel_size],
            )
            .with_padding(PaddingConfig2d::Explicit(self.padding, self.padding))
            .init(),
            bn: BatchNormConfig::new(self.c_out).init(),
            act: self.act.init(),
        }
    }

    pub fn init_with<B: Backend>(self, record: ConvBlockRecord<B>) -> ConvBlock<B> {
        ConvBlock {
            conv: Conv2dConfig::new(
                [self.c_in, self.c_out],
                [self.kernel_size, self.kernel_size],
            )
            .with_padding(PaddingConfig2d::Explicit(self.padding, self.padding))
            .init_with(record.conv),
            bn: BatchNormConfig::new(self.c_out).init_with(record.bn),
            act: self.act.init(),
        }
    }
}

#[derive(Debug, Module)]
pub struct ConvNBlock<B: Backend> {
    conv_a: ConvBlock<B>,
    conv_b: ConvBlock<B>,
}

impl<B: Backend> ConvNBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv_a.forward(x);
        let x = self.conv_b.forward(x);
        x
    }
}

#[derive(Config, Copy)]
pub struct ConvNBlockConfig {
    conv_a: ConvBlockConfig,
    conv_b: ConvBlockConfig,
}

impl ConvNBlockConfig {
    pub fn init<B: Backend>(&self) -> ConvNBlock<B> {
        ConvNBlock {
            conv_a: self.conv_a.init(),
            conv_b: self.conv_b.init(),
        }
    }

    pub fn init_with<B: Backend>(&self, record: ConvNBlockRecord<B>) -> ConvNBlock<B> {
        ConvNBlock {
            conv_a: self.conv_a.init_with(record.conv_a),
            conv_b: self.conv_b.init_with(record.conv_b),
        }
    }
}

#[derive(Debug, Module)]
pub struct ConvPoolBlock<B: Backend> {
    pool: MaxPool2d,
    conv: ConvBlock<B>,
}

impl<B: Backend> ConvPoolBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.pool.forward(x);
        x
    }
}

#[derive(Config)]
pub struct ConvPoolBlockConfig {
    pool: MaxPool2dConfig,
    conv: ConvBlockConfig,
}

impl ConvPoolBlockConfig {
    pub fn init<B: Backend>(&self) -> ConvPoolBlock<B> {
        ConvPoolBlock {
            pool: self.pool.init(),
            conv: self.conv.init(),
        }
    }

    pub fn init_with<B: Backend>(&self, record: ConvPoolBlockRecord<B>) -> ConvPoolBlock<B> {
        ConvPoolBlock {
            pool: self.pool.init(),
            conv: self.conv.init_with(record.conv),
        }
    }
}

#[derive(Config)]
pub struct InceptionBlockConfig {
    pub conv_1: ConvBlockConfig,
    pub conv_3: ConvNBlockConfig,
    pub conv_5: ConvNBlockConfig,
    pub conv_pool: ConvPoolBlockConfig,
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
            conv_1: ConvBlockConfig::new(in_c, conv_1_out, 1, 0, act),
            conv_3: ConvNBlockConfig::new(
                ConvBlockConfig::new(in_c, conv_3_reduction, 1, 0, act),
                ConvBlockConfig::new(conv_3_reduction, conv_3_out, 3, 1, act),
            ),
            conv_5: ConvNBlockConfig::new(
                ConvBlockConfig::new(in_c, conv_5_reduction, 1, 0, act),
                ConvBlockConfig::new(conv_5_reduction, conv_5_out, 5, 2, act),
            ),
            conv_pool: ConvPoolBlockConfig::new(
                MaxPool2dConfig::new([3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .with_strides([1, 1]),
                ConvBlockConfig::new(in_c, conv_pool_out, 1, 0, act),
            ),
        }
    }

    pub fn init<B: Backend>(&self) -> InceptionBlock<B> {
        InceptionBlock {
            conv_1: self.conv_1.init(),
            conv_3: self.conv_3.init(),
            conv_5: self.conv_5.init(),
            conv_pool: self.conv_pool.init(),
        }
    }

    pub fn init_with<B: Backend>(&self, record: InceptionBlockRecord<B>) -> InceptionBlock<B> {
        InceptionBlock {
            conv_1: self.conv_1.init_with(record.conv_1),
            conv_3: self.conv_3.init_with(record.conv_3),
            conv_5: self.conv_5.init_with(record.conv_5),
            conv_pool: self.conv_pool.init_with(record.conv_pool),
        }
    }
}

#[derive(Debug, Module)]
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

#[derive(Config)]
pub struct ModelInceptionModuleConfig {
    inception_1: Vec<InceptionBlockConfig>,
    pool_1: MaxPool2dConfig,
    inception_2: Vec<InceptionBlockConfig>,
    pool_2: MaxPool2dConfig,
    inception_3: Vec<InceptionBlockConfig>,
}

impl ModelInceptionModuleConfig {
    pub fn init<B: Backend>(self) -> ModelInceptionModule<B> {
        ModelInceptionModule {
            inception_1: self
                .inception_1
                .iter()
                .map(|config| config.init())
                .collect(),
            pool_1: self.pool_1.init(),
            inception_2: self
                .inception_2
                .iter()
                .map(|config| config.init())
                .collect(),
            pool_2: self.pool_1.init(),
            inception_3: self
                .inception_3
                .iter()
                .map(|config| config.init())
                .collect(),
        }
    }

    pub fn init_with<B: Backend>(
        self,
        record: ModelInceptionModuleRecord<B>,
    ) -> ModelInceptionModule<B> {
        ModelInceptionModule {
            inception_1: izip!(&self.inception_1, record.inception_1)
                .map(|(config, record)| config.init_with(record))
                .collect(),
            pool_1: self.pool_1.init(),
            inception_2: izip!(&self.inception_2, record.inception_2)
                .map(|(config, record)| config.init_with(record))
                .collect(),
            pool_2: self.pool_2.init(),
            inception_3: izip!(&self.inception_3, record.inception_3)
                .map(|(config, record)| config.init_with(record))
                .collect(),
        }
    }
}

#[derive(Debug, Module)]
pub struct ModelInceptionModule<B: Backend> {
    inception_1: Vec<InceptionBlock<B>>,
    pool_1: MaxPool2d,
    inception_2: Vec<InceptionBlock<B>>,
    pool_2: MaxPool2d,
    inception_3: Vec<InceptionBlock<B>>,
}

impl<B: Backend> ModelInceptionModule<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = x;
        for layer in &self.inception_1 {
            x = layer.forward(x);
        }
        x = self.pool_1.forward(x);
        for layer in &self.inception_2 {
            x = layer.forward(x);
        }
        x = self.pool_2.forward(x);
        for layer in &self.inception_3 {
            x = layer.forward(x);
        }
        x
    }
}

#[derive(Debug, Module)]
pub struct ModelOutputModule<B: Backend> {
    pool: AdaptiveAvgPool2d,
    linear: Linear<B>,
}

impl<B: Backend> ModelOutputModule<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.pool.forward(x);
        let x: Tensor<B, 2> = x.flatten(1, 3);
        let x = self.linear.forward(x);
        x
    }
}

#[derive(Config)]
pub struct ModelOutputModuleConfig {
    pub pool: AdaptiveAvgPool2dConfig,
    pub linear: LinearConfig,
}

impl ModelOutputModuleConfig {
    pub fn init<B: Backend>(self) -> ModelOutputModule<B> {
        ModelOutputModule {
            pool: self.pool.init(),
            linear: self.linear.init(),
        }
    }

    pub fn init_with<B: Backend>(self, record: ModelOutputModuleRecord<B>) -> ModelOutputModule<B> {
        ModelOutputModule {
            pool: self.pool.init(),
            linear: self.linear.init_with(record.linear),
        }
    }
}

#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    input: ConvBlock<B>,
    inception: ModelInceptionModule<B>,
    output: ModelOutputModule<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.input.forward(x);
        let x = self.inception.forward(x);
        let x = self.output.forward(x);
        x
    }
}

impl<B: Backend> ClassificationModel<B> for Model<B> {
    fn forward_classification(
        &self,
        x: Tensor<B, 4>,
        label: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let out = self.forward(x);
        let loss = CrossEntropyLoss::new(None).forward(out.clone(), label.clone());
        ClassificationOutput::new(loss, out, label)
    }
}

#[derive(Config)]
pub struct ModelConfig {
    input: ConvBlockConfig,
    inception: ModelInceptionModuleConfig,
    output: ModelOutputModuleConfig,
}

impl ModelConfig {
    pub fn init<B: Backend>(self) -> Model<B> {
        Model {
            input: self.input.init(),
            inception: self.inception.init(),
            output: self.output.init(),
        }
    }

    pub fn init_with<B: Backend>(self, record: ModelRecord<B>) -> Model<B> {
        Model {
            input: self.input.init_with(record.input),
            inception: self.inception.init_with(record.inception),
            output: self.output.init_with(record.output),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        let act = ActivationConfig::ReLU;
        let pool = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1));

        ModelConfig::new(
            ConvBlockConfig::new(3, 64, 3, 1, act),
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
                LinearConfig::new(128, NUM_CLASSES as _),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{backend::Backend, Float, Tensor};

    use crate::{
        data::cifar10::{IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES},
        model::inception::ModelConfig,
    };

    fn test<B: Backend>() {
        let model = ModelConfig::default().init::<B>();
        let input: Tensor<B, 4, Float> = Tensor::zeros([
            8,
            IMG_CHANNELS as usize,
            IMG_HEIGHT as usize,
            IMG_WIDTH as usize,
        ]);
        let output = model.forward(input);
        assert_eq!(output.shape().dims, [8, NUM_CLASSES as usize]);
    }

    #[test]
    fn sanity_ndarray() {
        test::<burn::backend::NdArray>();
    }

    #[test]
    fn sanity_tch_cpu() {
        test::<burn::backend::LibTorch>();
    }
}
