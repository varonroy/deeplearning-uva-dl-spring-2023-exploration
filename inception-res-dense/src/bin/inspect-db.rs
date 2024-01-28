use burn::data::dataset::Dataset;
use dotenv::dotenv;
use inception_res_dense as source;
use source::{
    data::cifar10::{self, CLASSES},
    utils::{buffer_to_image, show_image_terminal_color, Stats},
};

use anyhow::Result;
fn main() -> Result<()> {
    dotenv()?;

    // load the data
    let dataset = cifar10::Cifar10Dataset::train()?;
    println!("num items: {}", dataset.len());

    // get some images
    let images = [
        dataset.get(0).unwrap(),
        dataset.get(1).unwrap(),
        dataset.get(2).unwrap(),
        dataset.get(3).unwrap(),
    ];

    // sample an image
    let image = &images[1];
    println!("label: {} - {}", image.label, CLASSES[image.label]);
    show_image_terminal_color(&image.img);
    let img = buffer_to_image(&image.img);
    img.save("./out/example-image.png")?;

    // calcualte stats
    let stats = Stats::from_iter(dataset.iter().map(|item| item.img).into_iter());
    println!("{:#?}", stats);

    Ok(())
}
