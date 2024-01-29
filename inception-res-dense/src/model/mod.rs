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
