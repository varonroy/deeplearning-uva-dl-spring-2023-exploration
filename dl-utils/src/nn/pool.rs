use burn::{
    module::ConstantRecord,
    nn::pool::{
        AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, AvgPool2d, AvgPool2dConfig, MaxPool2d,
        MaxPool2dConfig,
    },
};

use crate::InitFns;

impl InitFns for AdaptiveAvgPool2dConfig {
    type Module = AdaptiveAvgPool2d;
    type Record = ConstantRecord;

    fn init(&self) -> Self::Module {
        self.init()
    }

    fn init_with(&self, _record: ConstantRecord) -> Self::Module {
        self.init()
    }
}

impl InitFns for AvgPool2dConfig {
    type Module = AvgPool2d;
    type Record = ConstantRecord;

    fn init(&self) -> Self::Module {
        self.init()
    }

    fn init_with(&self, _record: ConstantRecord) -> Self::Module {
        self.init()
    }
}

impl InitFns for MaxPool2dConfig {
    type Module = MaxPool2d;
    type Record = ConstantRecord;

    fn init(&self) -> Self::Module {
        burn::nn::pool::MaxPool2dConfig::init(self)
    }

    fn init_with(&self, _record: Self::Record) -> Self::Module {
        self.init()
    }
}
