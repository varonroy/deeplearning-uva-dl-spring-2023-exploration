// TODO
// use std::usize;
//
// use burn::{
//     config::Config,
//     module::Module,
//     nn::{
//         conv::{Conv2d, Conv2dConfig},
//         loss::CrossEntropyLoss,
//         pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
//         BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d,
//     },
//     tensor::{backend::Backend, Int, Tensor},
//     train::ClassificationOutput,
// };
//
// use crate::{res, sequential};
//
// use super::super::{Activation, ActivationConfig};
//
// sequential!(
//     ResBlockCore,
//     ResBlockCoreConfig,
//     ResBlockCoreRecord,
//     4 => 4,
//     (conv_1, Conv2d<B>, Conv2dConfig),
//     (bn_1, BatchNorm<B, 2>, BatchNormConfig),
//     (act_1, Activation, ActivationConfig),
//     (conv_2, Conv2d<B>, Conv2dConfig),
//     (bn_2, BatchNorm<B, 2>, BatchNormConfig)
// );
//
// impl ResBlockCoreConfig {
//     pub fn new2(c_in: usize, c_out: usize, subsample: bool, act: ActivationConfig) -> Self {
//         Self::new(
//             Conv2dConfig::new([c_in, c_out], [3, 3])
//                 .with_padding(PaddingConfig2d::Explicit(1, 1))
//                 .with_stride(if subsample { [2, 2] } else { [1, 1] })
//                 .with_bias(false),
//             BatchNormConfig::new(c_out),
//             act,
//             Conv2dConfig::new([c_out, c_out], [3, 3])
//                 .with_padding(PaddingConfig2d::Explicit(1, 1))
//                 .with_stride([1, 1])
//                 .with_bias(false),
//             BatchNormConfig::new(c_out),
//         )
//     }
// }
//
// #[derive(Debug, Module)]
// pub struct ResBlock<B: Backend> {
//     core: ResBlockCore<B>,
//     conv: Option<Conv2d<B>>,
//     act: Activation,
// }
//
// impl<B: Backend> ResBlock<B> {
//     pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
//         let z = self.core.forward(x.clone());
//         let z = if let Some(conv) = &self.conv {
//             conv.forward(z)
//         } else {
//             z
//         };
//         let x = x + z;
//         let x = self.act.forward(x);
//         x
//     }
// }
//
// #[derive(Config)]
// pub struct ResBlockConfig {
//     pub core: ResBlockCoreConfig,
//     pub conv: Option<Conv2dConfig>,
//     pub act: ActivationConfig,
// }
//
// impl ResBlockConfig {
//     pub fn init<B: Backend>(&self) -> ResBlock<B> {
//         ResBlock {
//             core: self.core.init(),
//             conv: self.conv.as_ref().map(|config| config.init()),
//             act: self.act.init(),
//         }
//     }
//
//     pub fn init_with<B: Backend>(&self, record: ResBlockRecord<B>) -> ResBlock<B> {
//         ResBlock {
//             core: self.core.init_with(record.core),
//             conv: self
//                 .conv
//                 .as_ref()
//                 .zip(record.conv)
//                 .map(|(config, record)| config.init_with(record)),
//             act: self.act.init(),
//         }
//     }
// }
