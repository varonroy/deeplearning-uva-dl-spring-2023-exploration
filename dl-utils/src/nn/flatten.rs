use burn::config::Config;
use burn::module::{ConstantRecord, Module};
use burn::tensor::{backend::Backend, Tensor};

use crate::InitFns;

#[derive(Debug, Module, Clone, Copy)]
pub struct Flatten<const DI: usize, const DO: usize> {
    pub start_dim: usize,
    pub end_dim: usize,
}

impl<const DI: usize, const DO: usize> Flatten<DI, DO> {
    pub fn forward<B: Backend>(&self, x: Tensor<B, DI>) -> Tensor<B, DO> {
        x.flatten(self.start_dim, self.end_dim)
    }
}

macro_rules! impl_flatten {
    ($config_name:ident, $dim_in:literal, $dim_out:literal) => {
        #[derive(Debug, Config)]
        pub struct $config_name {
            pub start_dim: usize,
            pub end_dim: usize,
        }

        impl InitFns for $config_name {
            type Module = Flatten<$dim_in, $dim_out>;
            type Record = ConstantRecord;

            fn init(&self) -> Self::Module {
                Flatten {
                    start_dim: self.start_dim,
                    end_dim: self.end_dim,
                }
            }

            fn init_with(&self, _record: Self::Record) -> Self::Module {
                Self::Module {
                    start_dim: self.start_dim,
                    end_dim: self.end_dim,
                }
            }
        }
    };
}

impl_flatten!(Flatten21Config, 2, 1);
impl_flatten!(Flatten31Config, 3, 1);
impl_flatten!(Flatten32Config, 3, 2);
impl_flatten!(Flatten41Config, 4, 1);
impl_flatten!(Flatten42Config, 4, 2);
impl_flatten!(Flatten43Config, 4, 3);
impl_flatten!(Flatten51Config, 5, 1);
impl_flatten!(Flatten52Config, 5, 2);
impl_flatten!(Flatten53Config, 5, 3);
impl_flatten!(Flatten54Config, 5, 4);
