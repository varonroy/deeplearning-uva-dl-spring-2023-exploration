use anyhow::{Context, Result};

pub fn get_env(key: &str) -> Result<String> {
    std::env::var(key).context(format!("getting env varaible `{key}`"))
}

pub fn show_image_terminal<const C: usize, const W: usize, const H: usize>(
    img: &[[[f32; C]; W]; H],
) {
    fn eval_pixel<const C: usize>(pixel: &[f32; C]) -> &'static str {
        let c = pixel.into_iter().map(|x| *x as f64).sum::<f64>() / (C as f64);
        let c = (c * 4.0) as i32;
        match c {
            1 => ".",
            2 => "×",
            3 | 4 => "#",
            _ => " ",
        }
    }

    for i in 0..H {
        for j in 0..W {
            let c = eval_pixel(&img[i][j]);
            print!("{c}");
        }
        println!();
    }
}

pub fn show_image_terminal_color<const C: usize, const W: usize, const H: usize>(
    img: &[[[f32; C]; W]; H],
) {
    for i in 0..H {
        for j in 0..W {
            let p = &img[i][j];
            let color = termion::color::Rgb(
                (p[0] * 255.0) as _,
                (p[1] * 255.0) as _,
                (p[2] * 255.0) as _,
            );
            print!("{} ", termion::color::Bg(color));
        }
        println!("{}", termion::color::Bg(termion::color::Reset));
    }
}

pub fn buffer_to_image<const C: usize, const W: usize, const H: usize>(
    img: &[[[f32; C]; W]; H],
) -> image::DynamicImage {
    let mut imgbuf = image::ImageBuffer::new(W as _, H as _);
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let i = y as usize;
        let j = x as usize;
        *pixel = image::Rgb([
            (img[i][j][0] * 255.0) as u8,
            (img[i][j][1] * 255.0) as u8,
            (img[i][j][2] * 255.0) as u8,
        ]);
    }
    imgbuf.into()
}

#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub dim: [usize; 3],
    pub value_range: [f32; 2],
    pub mean: f32,
    pub var: f32,
    pub stddev: f32,
}

impl Stats {
    pub fn from_iter<const C: usize, const W: usize, const H: usize>(
        iter: impl Iterator<Item = [[[f32; C]; W]; H]>,
    ) -> Self {
        let mut s = 0.0;
        let mut s2 = 0.0;
        let mut count = 0;
        let mut value_range = [0.0f32, 0.0f32];
        let d = (C * W * H) as f32;
        for item in iter {
            item.into_iter().flatten().flatten().for_each(|x| {
                value_range[0] = value_range[0].min(x);
                value_range[1] = value_range[1].max(x);
            });

            s += item.into_iter().flatten().flatten().sum::<f32>() / d;
            s2 += item
                .into_iter()
                .flatten()
                .flatten()
                .map(|x| x * x)
                .sum::<f32>()
                / d;
            count += 1;
        }
        let count_f32 = count as f32;
        let mean = s / count_f32;
        let var = (s2 / count_f32) - mean * mean;
        let stddev = var.sqrt();
        Self {
            dim: [H, W, C],
            value_range,
            mean,
            var,
            stddev,
        }
    }
}
