use burn::{
    nn::{PositionalEncoding, PositionalEncodingConfig, PositionalEncodingRecord},
    tensor::backend::Backend,
};

use crate::InitFnsB;

impl<B: Backend> InitFnsB<B> for PositionalEncodingConfig {
    type Module = PositionalEncoding<B>;
    type Record = PositionalEncodingRecord<B>;

    fn init(&self) -> Self::Module {
        self.init()
    }

    fn init_with(&self, _record: Self::Record) -> Self::Module {
        self.init()
    }
}
