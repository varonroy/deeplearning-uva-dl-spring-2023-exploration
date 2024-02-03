use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, ReLU,
    },
    tensor::{backend::Backend, Tensor},
};
use dl_utils::nn::{Flatten, Flatten42Config};
use dl_utils_burn_sequential::SequentialForward;

type Flatten42 = Flatten<4, 2>;

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 4)]
pub struct ConvBlock<B: Backend> {
    pub conv: Conv2d<B>,
    pub bn: BatchNorm<B, 2>,
    pub act: ReLU,
}

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 4)]
pub struct ConvPoolBlock<B: Backend> {
    pool: MaxPool2d,
    conv: ConvBlock<B>,
}

#[derive(Debug, Module, SequentialForward)]
#[dims(4, 2)]
struct MyModule<B: Backend> {
    bn: BatchNorm<B, 2>,
    block: ConvPoolBlock<B>,
    tuple: (Vec<Conv2d<B>>, BatchNorm<B, 2>, Flatten42, ReLU),
    dropout: Dropout,
    linear: Linear<B>,
}

#[test]
fn simple() {
    type B = burn::backend::NdArray;
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let config = MyModuleConfig {
        bn: BatchNormConfig::new(3),
        block: ConvPoolBlockConfig::new(
            MaxPool2dConfig::new([2, 2]),
            ConvBlockConfig::new(
                Conv2dConfig::new([3, 3], [3, 3]).with_stride([2, 2]),
                BatchNormConfig::new(3),
                (),
            ),
        ),
        tuple: (
            vec![
                Conv2dConfig::new([3, 3], [2, 2]),
                Conv2dConfig::new([3, 3], [2, 2]),
            ],
            BatchNormConfig::new(3),
            Flatten42Config::new(1, 3),
            (),
        ),
        dropout: DropoutConfig::new(0.5),
        linear: LinearConfig::new(3, 1),
    };

    let x: Tensor<B, 4> = Tensor::random(
        [1, 3, 8, 8],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    println!("{:?}", x.shape());

    let net = config.init::<B>(&device);
    let net_with = config.init_with(net.clone().into_record());

    let x1 = net.forward(x.clone());
    let x2 = net_with.forward(x.clone());

    println!("{}", x1);
    println!("{}", x2);
}
