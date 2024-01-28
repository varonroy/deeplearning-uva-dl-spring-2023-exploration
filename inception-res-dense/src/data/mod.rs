use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
};

use self::cifar10::Cifar10Item;

pub mod cifar10;

pub struct Cifar10Batcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Cifar10Batcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Debug, Clone)]
pub struct Cifar10Batch<B: Backend> {
    pub img: Tensor<B, 4>,
    pub label: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<Cifar10Item, Cifar10Batch<B>> for Cifar10Batcher<B> {
    fn batch(&self, items: Vec<Cifar10Item>) -> Cifar10Batch<B> {
        let img = items
            .iter()
            .map(|item| Data::<f32, 3>::from(item.img))
            .map(|data| Tensor::<B, 3>::from_data(data.convert()))
            .map(|tensor|
                 // H x W x C -> C x W x H
                 tensor.swap_dims(0, 2)
                 // H x W x C -> C x H x W
                 .transpose())
            // normalize
            .map(|tensor| (tensor - 0.5) / 0.2)
            .collect();

        let label = items
            .iter()
            .map(|item| item.label as i64)
            .map(|label| Data::from([label.elem()]))
            .map(|data| Tensor::<B, 1, Int>::from_data(data))
            .collect();

        let img = Tensor::stack(img, 0).to_device(&self.device);
        let label = Tensor::cat(label, 0).to_device(&self.device);

        Cifar10Batch { img, label }
    }
}
