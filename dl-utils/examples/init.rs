use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::backend::Backend,
};
use dl_utils::InitFns;

#[derive(Debug, Module)]
struct MyModule<B: Backend> {
    l: Linear<B>,
}

struct MyModuleConfig {
    l: LinearConfig,
}

// impl<B: Backend> InitFns for MyModuleConfig {
//     type Module = MyModule<B>;
//     type Record = MyModuleRecord<B>;
//
//     fn init(&self) -> Self::Module {
//         todo!()
//     }
//
//     fn init_with(&self, record: Self::Record) -> Self::Module {
//         todo!()
//     }
// }

fn main() {}
