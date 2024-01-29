use anyhow::Context;
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::{AutodiffBackend, Backend},
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
    },
};

use crate::{
    data::{cifar10::Cifar10Dataset, Cifar10Batch, Cifar10Batcher},
    model::ClassificationModel,
    utils::get_env,
};

macro_rules! impl_steps {
    ($model:ty) => {
        impl<B: AutodiffBackend> TrainStep<Cifar10Batch<B>, ClassificationOutput<B>> for $model {
            fn step(&self, batch: Cifar10Batch<B>) -> TrainOutput<ClassificationOutput<B>> {
                let item = self.forward_classification(batch.img, batch.label);

                TrainOutput::new(self, item.loss.backward(), item)
            }
        }

        impl<B: Backend> ValidStep<Cifar10Batch<B>, ClassificationOutput<B>> for $model {
            fn step(&self, batch: Cifar10Batch<B>) -> ClassificationOutput<B> {
                self.forward_classification(batch.img, batch.label)
            }
        }
    };
}

impl_steps!(crate::model::inception::Model<B>);
impl_steps!(crate::model::res::pre::Model<B>);
impl_steps!(crate::model::res::regular::Model<B>);
impl_steps!(crate::model::dense::Model<B>);

#[derive(Config)]
pub struct TrainingConfig {
    // pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(config: TrainingConfig, device: B::Device) -> anyhow::Result<()> {
    log::info!("- train -");
    // let model = crate::model::inception::ModelConfig::default().init();
    // let model = crate::model::res::regular::ModelConfig::default().init();
    // let model = crate::model::res::pre::ModelConfig::default().init();
    let model = crate::model::dense::ModelConfig::default().init();

    let artifacts_dir = get_env("ARTIFACTS_DIR")?;

    log::info!("loading datastes");
    let dataset_train = Cifar10Dataset::train().context("train dataset")?;
    let dataset_test = Cifar10Dataset::test().context("test dataset")?;

    log::info!("creating batchers");
    let batcher_train = Cifar10Batcher::<B>::new(device.clone());
    let batcher_test = Cifar10Batcher::<B::InnerBackend>::new(device.clone());

    log::info!("creating loaders");
    let loader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(0)
        .num_workers(config.num_workers)
        .build(dataset_train);
    let loader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(0)
        .num_workers(config.num_workers)
        .build(dataset_test);

    log::info!("creating learner");
    let learner = LearnerBuilder::new(&artifacts_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(model, config.optimizer.init(), config.learning_rate);

    log::info!("fitting");
    let trained_model = learner.fit(loader_train, loader_test);

    let model_path = "./out/trained-model";
    log::info!("saving to {model_path}");
    trained_model
        .save_file(model_path, &CompactRecorder::new())
        .context("saving trained model")?;

    Ok(())
}
