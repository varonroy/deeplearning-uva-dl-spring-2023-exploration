use std::usize;

use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        loss::CrossEntropyLoss,
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d,
    },
    tensor::{backend::Backend, Int, Tensor},
    train::ClassificationOutput,
};

use crate::{conv2d, data::cifar10::NUM_CLASSES, model::ClassificationModel, sequential};

use super::super::{Activation, ActivationConfig, Flatten, FlattenConfig42};

sequential!(
    ResBlockCore,
    ResBlockCoreConfig,
    ResBlockCoreRecord,
    4 => 4,
    (bn_1, BatchNorm<B, 2>, BatchNormConfig),
    (act_1, Activation, ActivationConfig),
    (conv_1, Conv2d<B>, Conv2dConfig),
    (bn_2, BatchNorm<B, 2>, BatchNormConfig),
    (act_2, Activation, ActivationConfig),
    (conv_2, Conv2d<B>, Conv2dConfig)
);

impl ResBlockCoreConfig {
    pub fn new2(c_in: usize, c_out: usize, subsample: bool, act: ActivationConfig) -> Self {
        Self::new(
            BatchNormConfig::new(c_in),
            act,
            Conv2dConfig::new([c_in, c_out], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_stride(if subsample { [2, 2] } else { [1, 1] })
                .with_bias(false),
            BatchNormConfig::new(c_out),
            act,
            Conv2dConfig::new([c_out, c_out], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_stride([1, 1])
                .with_bias(false),
        )
    }
}

#[derive(Debug, Module)]
pub struct Downsample<B: Backend> {
    pub bn: BatchNorm<B, 2>,
    pub act: Activation,
    pub conv: Conv2d<B>,
}

impl<B: Backend> Downsample<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.bn.forward(x);
        let x = self.act.forward(x);
        self.conv.forward(x)
    }
}

#[derive(Debug, Config)]
pub struct DownsampleConfig {
    pub bn: BatchNormConfig,
    pub act: ActivationConfig,
    pub conv: Conv2dConfig,
}

impl DownsampleConfig {
    pub fn new2(c_in: usize, c_out: usize, act: ActivationConfig) -> Self {
        Self::new(
            BatchNormConfig::new(c_in),
            act,
            conv2d!(
                c_in = c_in,
                c_out = c_out,
                kernel_size = 1,
                padding = 0,
                stride = 1,
                bias = false,
            ),
        )
    }

    pub fn init<B: Backend>(&self) -> Downsample<B> {
        Downsample {
            bn: self.bn.init(),
            act: self.act.init(),
            conv: self.conv.init(),
        }
    }

    pub fn init_with<B: Backend>(&self, record: DownsampleRecord<B>) -> Downsample<B> {
        Downsample {
            bn: self.bn.init_with(record.bn),
            act: self.act.init_with(record.act),
            conv: self.conv.init_with(record.conv),
        }
    }
}

#[derive(Debug, Module)]
pub struct ResBlock<B: Backend> {
    pub core: ResBlockCore<B>,
    pub downsample: Option<Downsample<B>>,
}

impl<B: Backend> ResBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.core.forward(x.clone())
            + self
                .downsample
                .as_ref()
                .map(|layer| layer.forward(x.clone()))
                .unwrap_or(x.clone())
    }
}

#[derive(Config)]
pub struct ResBlockConfig {
    pub core: ResBlockCoreConfig,
    pub downsample: Option<DownsampleConfig>,
}

impl ResBlockConfig {
    pub fn new2(c_in: usize, c_out: usize, subsample: bool, act: ActivationConfig) -> Self {
        Self {
            core: ResBlockCoreConfig::new2(c_in, c_out, subsample, act),
            downsample: subsample.then_some(DownsampleConfig::new2(c_in, c_out, act)),
        }
    }

    pub fn init<B: Backend>(&self) -> ResBlock<B> {
        ResBlock {
            core: self.core.init(),
            downsample: self.downsample.as_ref().map(|config| config.init()),
        }
    }

    pub fn init_with<B: Backend>(&self, record: ResBlockRecord<B>) -> ResBlock<B> {
        ResBlock {
            core: self.core.init_with(record.core),
            downsample: self
                .downsample
                .as_ref()
                .zip(record.downsample)
                .map(|(config, record)| config.init_with(record)),
        }
    }
}

#[derive(Debug, Module)]
pub struct ResNetBody<B: Backend> {
    blocks: Vec<ResBlock<B>>,
}

impl<B: Backend> ResNetBody<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = x;
        for layer in &self.blocks {
            x = layer.forward(x);
        }
        x
    }
}

#[derive(Config)]
pub struct ResNetBodyConfig {
    pub blocks: Vec<ResBlockConfig>,
}

impl ResNetBodyConfig {
    pub fn init<B: Backend>(&self) -> ResNetBody<B> {
        ResNetBody {
            blocks: self.blocks.iter().map(|config| config.init()).collect(),
        }
    }

    pub fn init_with<B: Backend>(&self, record: ResNetBodyRecord<B>) -> ResNetBody<B> {
        ResNetBody {
            blocks: self
                .blocks
                .iter()
                .zip(record.blocks.into_iter())
                .map(|(config, record)| config.init_with(record))
                .collect(),
        }
    }
}

use super::super::InitWith;
sequential!(
    ResNet,
    ResNetConfig,
    ResNetRecord,
    4 => 2,
    // input
    (input, Conv2d<B>, Conv2dConfig),
    // body
    (body, ResNetBody<B>, ResNetBodyConfig),
    // output
    (output_pool, AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig),
    (output_flatten, Flatten<4, 2>, FlattenConfig42),
    (output_linear, Linear<B>, LinearConfig)
);

impl Default for ResNetConfig {
    fn default() -> Self {
        let act = ActivationConfig::ReLU;
        let num_blocks = [3, 3, 3];
        let c_hidden = [16, 32, 64];

        Self::new(
            conv2d!(
                c_in = 3,
                c_out = c_hidden[0],
                kernel_size = 3,
                padding = 1,
                stride = 1,
                bias = false,
            ),
            ResNetBodyConfig::new(
                num_blocks
                    .into_iter()
                    .enumerate()
                    .flat_map(|(block_idx, block_count)| {
                        (0..block_count).map(move |bc| {
                            let subsample = bc == 0 && block_idx > 0;
                            ResBlockConfig::new2(
                                c_hidden[if subsample { block_idx - 1 } else { block_idx }],
                                c_hidden[block_idx],
                                subsample,
                                act,
                            )
                        })
                    })
                    .collect(),
            ),
            AdaptiveAvgPool2dConfig::new([1, 1]),
            FlattenConfig42::new(1, 3),
            LinearConfig::new(*c_hidden.last().unwrap(), NUM_CLASSES as _),
        )
    }
}

impl<B: Backend> ClassificationModel<B> for ResNet<B> {
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

#[cfg(test)]
mod tests {
    use burn::tensor::{backend::Backend, Float, Tensor};

    use super::ResNetConfig;
    use crate::data::cifar10::{IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES};

    fn test<B: Backend>() {
        let model = ResNetConfig::default().init::<B>();
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
