

use std::usize;

use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        loss::CrossEntropyLoss,
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        BatchNorm, BatchNormConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, Int, Tensor},
    train::ClassificationOutput,
};
use dl_utils_burn_sequential::SequentialForward;

use crate::{data::cifar10::NUM_CLASSES, model::ClassificationModel};

use dl_utils::conv_2d;
use dl_utils::nn::{Activation, ActivationConfig, Flatten, Flatten42Config};
use dl_utils::InitFns;

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 4)]
pub struct ResBlockCore<B: Backend> {
    conv_1: Conv2d<B>,
    bn_1: BatchNorm<B, 2>,
    act_1: Activation,
    conv_2: Conv2d<B>,
    bn_2: BatchNorm<B, 2>,
}

impl ResBlockCoreConfig {
    pub fn new2(c_in: usize, c_out: usize, subsample: bool, act: ActivationConfig) -> Self {
        let c_out = if !subsample { c_in } else { c_out };

        Self::new(
            conv_2d!(
                c_in,
                c_out,
                kernel_size = 3,
                padding = 1,
                stride = if subsample { 2 } else { 1 },
                bias = false
            ),
            BatchNormConfig::new(c_out),
            act,
            conv_2d!(c_out, c_out, kernel_size = 3, padding = 1, bias = false),
            BatchNormConfig::new(c_out),
        )
    }
}

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 4)]
pub struct Downsample<B: Backend> {
    conv: Conv2d<B>,
}

impl DownsampleConfig {
    pub fn new2(c_in: usize, c_out: usize) -> Self {
        Self::new(conv_2d!(c_in, c_out, kernel_size = 1, stride = 2, bias = false))
    }
}

#[derive(Debug, Module, SequentialForward)]
#[manual_forward]
pub struct ResBlock<B: Backend> {
    pub core: ResBlockCore<B>,
    pub downsample: Option<Downsample<B>>,
    pub act: Activation,
}

impl<B: Backend> ResBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.act.forward(self.core.forward(x.clone())
            + self
                .downsample
                .as_ref()
                .map(|layer| layer.forward(x.clone()))
                .unwrap_or(x.clone()))
    }
}

impl ResBlockConfig {
    pub fn new2(c_in: usize, c_out: usize, subsample: bool, act: ActivationConfig) -> Self {
        Self {
            core: ResBlockCoreConfig::new2(c_in, c_out, subsample, act),
            downsample: subsample.then_some(DownsampleConfig::new2(c_in, c_out)),
            act,
        }
    }
}

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 4)]
pub struct ResNetBody<B: Backend> {
    blocks: Vec<ResBlock<B>>,
}

type Flatten42 = Flatten<4, 2>;

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 2)]
pub struct Model<B: Backend> {
    input: Conv2d<B>,
    body: ResNetBody<B>,
    output_pool: AdaptiveAvgPool2d,
    output_flatten: Flatten42,
    output_linear: Linear<B>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        let act = ActivationConfig::ReLU;
        let num_blocks = [3, 3, 3];
        let c_hidden = [16, 32, 64];

        Self::new(
            conv_2d!(
                3,
                c_hidden[0],
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
            Flatten42Config::new(1, 3),
            LinearConfig::new(*c_hidden.last().unwrap(), NUM_CLASSES as _),
        )
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

#[cfg(test)]
mod tests {
    use burn::tensor::{backend::Backend, Float, Tensor};

    use super::ModelConfig;
    use crate::data::cifar10::{IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES};

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
