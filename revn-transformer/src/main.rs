use burn::{
    backend::{libtorch::LibTorchDevice, Autodiff, LibTorch},
    optim::AdamConfig,
};
use dotenv::dotenv;

use crate::train::train;

mod data;
mod model;
mod train;

fn main() -> anyhow::Result<()> {
    dotenv()?;

    let device = LibTorchDevice::Cuda(0);
    println!("setup device");

    println!("starting training");
    train::<Autodiff<LibTorch<f32>>>(
        train::TrainingConfig::new(10, 10, 4096, AdamConfig::new()).with_num_epochs(10_000),
        device,
    )?;
    println!("done!");
    Ok(())
}
