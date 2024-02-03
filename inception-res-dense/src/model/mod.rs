use burn::{
    module::Module,
    tensor::{backend::Backend, Int, Tensor},
    train::ClassificationOutput,
};

pub mod dense;
pub mod inception;
pub mod res;

pub trait ForwardModule<B: Backend, const NI: usize, const NO: usize>: Module<B> {
    fn forward(&self, x: Tensor<B, NI>) -> Tensor<B, NO>;
}

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

#[cfg(test)]
mod tests {
    use super::dense;
    use super::inception;
    use super::res;
    use crate::data::cifar10::{IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES};
    use burn::tensor::{backend::Backend, Float, Tensor};

    macro_rules! test_model {
        ($model:ident, $device:ident) => {
            let input: Tensor<B, 4, Float> = Tensor::zeros(
                [
                    8,
                    IMG_CHANNELS as usize,
                    IMG_HEIGHT as usize,
                    IMG_WIDTH as usize,
                ],
                $device,
            );
            let output = $model.forward(input);
            assert_eq!(output.shape().dims, [8, NUM_CLASSES as usize]);
        };
    }

    fn test_models<B: Backend>(device: &B::Device) {
        let model = dense::ModelConfig::default().init::<B>(device);
        test_model!(model, device);

        let model = inception::ModelConfig::default().init::<B>(device);
        test_model!(model, device);

        let model = res::pre::ModelConfig::default().init::<B>(device);
        test_model!(model, device);

        let model = res::regular::ModelConfig::default().init::<B>(device);
        test_model!(model, device);
    }

    #[test]
    fn sanity_ndarray() {
        type B = burn::backend::ndarray::NdArray;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        test_models::<B>(&device);
    }

    #[test]
    fn sanity_tch_cpu() {
        type B = burn::backend::LibTorch;
        test_models::<B>(&burn::backend::libtorch::LibTorchDevice::Cpu);
    }
}
