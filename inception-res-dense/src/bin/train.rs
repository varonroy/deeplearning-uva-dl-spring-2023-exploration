use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{Autodiff, LibTorch};
use burn::optim::AdamConfig;
use dotenv::dotenv;
use inception_res_dense::training;

fn main() -> anyhow::Result<()> {
    dotenv()?;

    let device = LibTorchDevice::Cuda(0);
    println!("setup device");

    println!("starting training");
    training::train::<Autodiff<LibTorch<f32>>>(
        training::TrainingConfig::new(AdamConfig::new()),
        device,
    )?;
    println!("done!");
    Ok(())
}
