use burn::{
    module::ConstantRecord,
    nn::{Dropout, DropoutConfig},
};

use crate::InitFns;

impl InitFns for DropoutConfig {
    type Module = Dropout;
    type Record = ConstantRecord;

    fn init(&self) -> Self::Module {
        self.init()
    }

    fn init_with(&self, _record: ConstantRecord) -> Self::Module {
        self.init()
    }
}
