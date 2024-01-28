use std::path::PathBuf;

use burn::data::dataset::{
    transform::{Mapper, MapperDataset},
    Dataset, HuggingfaceDatasetLoader, SqliteDataset,
};
use image::GenericImageView;

use crate::utils::get_env;

pub const IMG_WIDTH: u32 = 32;
pub const IMG_HEIGHT: u32 = 32;
pub const IMG_CHANNELS: u32 = 3;
pub const NUM_CLASSES: u32 = 10;

pub const CLASSES: [&'static str; 10] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
];

// Full item structure:
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Cifar10RawItem {
    pub img_bytes: Vec<u8>,
    pub img_path: Option<PathBuf>,
    pub label: i64,
    pub row_id: i64,
}

#[derive(Debug, Clone)]
pub struct Cifar10Item {
    pub img: [[[f32; IMG_CHANNELS as _]; IMG_WIDTH as _]; IMG_HEIGHT as _],
    pub label: usize,
}

/// Taken from [burn](https://burn.dev/docs/src/burn_dataset/source/huggingface/mnist.rs.html#58).
/// Converts `RawItem`s to `Item`.
#[derive(Debug, Clone, Copy, Default)]
struct BytesToImage;

impl Mapper<Cifar10RawItem, Cifar10Item> for BytesToImage {
    fn map(&self, item: &Cifar10RawItem) -> Cifar10Item {
        let image = image::load_from_memory(&item.img_bytes).unwrap();

        debug_assert_eq!(image.dimensions(), (IMG_WIDTH, IMG_HEIGHT));

        let mut img = [[[0f32; IMG_CHANNELS as _]; IMG_WIDTH as _]; IMG_HEIGHT as _];
        for (x, y, pixel) in image.pixels() {
            let i = y as usize;
            let j = x as usize;

            img[i][j][0] = pixel[0] as f32 / 255.0;
            img[i][j][1] = pixel[1] as f32 / 255.0;
            img[i][j][2] = pixel[2] as f32 / 255.0;
        }

        Cifar10Item {
            img,
            label: item.label as _,
        }
    }
}

pub struct Cifar10Dataset {
    dataset: MapperDataset<SqliteDataset<Cifar10RawItem>, BytesToImage, Cifar10RawItem>,
}

impl Cifar10Dataset {
    pub fn train() -> anyhow::Result<Self> {
        Self::new("train")
    }

    pub fn test() -> anyhow::Result<Self> {
        Self::new("test")
    }

    fn new(split: &str) -> anyhow::Result<Self> {
        let base_dir = get_env("DB_BASE_DIR")?;
        Ok(Self {
            dataset: MapperDataset::new(
                HuggingfaceDatasetLoader::new("cifar10")
                    .with_base_dir(&base_dir)
                    .dataset(split)?,
                BytesToImage::default(),
            ),
        })
    }
}

impl Dataset<Cifar10Item> for Cifar10Dataset {
    fn get(&self, index: usize) -> Option<Cifar10Item> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
