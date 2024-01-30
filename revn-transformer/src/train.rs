use anyhow::Context;
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};

use crate::data::{MyBatcher, MyDataset};
use crate::model::ModelConfig;

#[derive(Config)]
pub struct TrainingConfig {
    pub num_classes: usize,
    pub seq_length: usize,
    pub dataset_size: usize,
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
    let model = ModelConfig::new2(config.num_classes).init();

    let artifacts_dir = std::env::var("ARTIFACTS_DIR").expect("ARTIFACTS_DIR is not set");
    assert!(!artifacts_dir.is_empty());

    log::info!("creating datasets");
    let dataset_train = MyDataset::new(config.num_classes, config.seq_length, config.dataset_size);
    let dataset_test = MyDataset::new(config.num_classes, config.seq_length, config.dataset_size);

    log::info!("creating batchers");
    let batcher_train = MyBatcher::<B>::new(device.clone());
    let batcher_test = MyBatcher::<B::InnerBackend>::new(device.clone());

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
