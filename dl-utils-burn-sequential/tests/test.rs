use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, Linear, LinearConfig, ReLU,
    },
    tensor::{backend::Backend, Float, Tensor},
};
use dl_utils::nn::{Flatten, Flatten42Config, ReLUConfig};
use dl_utils::InitFns;
use dl_utils_burn_sequential::SequentialForward;
use serde;

type Flatten42 = Flatten<4, 2>;

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 2)]
// #[res]
struct MyModule<B: Backend> {
    conv: Conv2d<B>,
    linear: Vec<Linear<B>>,
    layers: Vec<Option<Linear<B>>>,
    bn1: Option<BatchNorm<B, 2>>,
    flatten: Flatten42,
    act1: ReLU,
}

#[test]
fn simple() {
    type B = burn::backend::NdArray;

    let config = MyModuleConfig::new(
        Conv2dConfig::new([3, 1], [2, 2]),
        vec![],
        vec![],
        None,
        Flatten42Config::new(1, 3),
        ReLUConfig::new(),
    );
    let module = config.init::<B>();

    let _conv = &module.conv;
    let _layers = &module.layers;
    let _act1 = &module.act1;

    let x = Tensor::<B, 4, Float>::zeros([4, 3, 64, 64]);
    let _y: Tensor<B, 2, Float> = module.forward(x);
}
