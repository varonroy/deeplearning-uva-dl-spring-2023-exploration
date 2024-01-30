use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
};
use derive_new::new;
use itertools::Itertools;

fn one_hot(i: usize, len: usize) -> Vec<f32> {
    let mut v = vec![0.0; len];
    v[i] = 1.0;
    v
}

#[derive(Debug, Clone, new)]
pub struct MyItem {
    pub item: Vec<Vec<f32>>,
    pub index: Vec<usize>,
}

impl MyItem {
    fn rand(num_classes: usize, seq: usize) -> Self {
        let index = (0..seq)
            .into_iter()
            .map(|_| rand::random::<usize>() % num_classes)
            .collect_vec();
        let item = index.iter().map(|i| one_hot(*i, num_classes)).collect_vec();
        Self::new(item, index)
    }
}

pub struct MyDataset {
    items: Vec<MyItem>,
}

impl MyDataset {
    pub fn new(num_classes: usize, seq: usize, dataset_size: usize) -> Self {
        Self {
            items: (0..dataset_size)
                .into_iter()
                .map(|_| MyItem::rand(num_classes, seq))
                .collect(),
        }
    }
}

impl Dataset<MyItem> for MyDataset {
    fn get(&self, index: usize) -> Option<MyItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Debug, Clone)]
pub struct MyBatch<B: Backend> {
    pub item: Tensor<B, 3>,
    pub target: Tensor<B, 3>,
    pub label: Tensor<B, 2, Int>,
}

#[derive(new)]
pub struct MyBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> Batcher<MyItem, MyBatch<B>> for MyBatcher<B> {
    fn batch(&self, items: Vec<MyItem>) -> MyBatch<B> {
        let batch_size = items.len();
        let seq = items[0].item.len();
        let num_classes = items[0].item[0].len();

        let item = items
            .iter()
            .map(|item| item.item.clone().into_iter().flatten().collect_vec())
            .map(|item| Data::<f32, 2>::new(item, [seq, num_classes].into()))
            .map(|data| Tensor::<B, 2>::from_data(data.convert()))
            .collect();

        let target = items
            .iter()
            .map(|item| item.item.clone().into_iter().rev().flatten().collect_vec())
            .map(|item| Data::<f32, 2>::new(item, [seq, num_classes].into()))
            .map(|data| Tensor::<B, 2>::from_data(data.convert()))
            .collect();

        let label = items
            .iter()
            .map(|item| {
                Data::<i64, 1>::new(
                    item.index
                        .clone()
                        .into_iter()
                        .map(|x| (x as i64).elem())
                        .rev()
                        .collect_vec(),
                    [seq].into(),
                )
            })
            .map(|data| Tensor::<B, 1, Int>::from_data(data.convert()))
            .collect();

        let item = Tensor::stack(item, 0).to_device(&self.device);
        let target = Tensor::stack(target, 0).to_device(&self.device);
        let label = Tensor::stack(label, 0).to_device(&self.device);

        MyBatch {
            item,
            target,
            label,
        }
    }
}
