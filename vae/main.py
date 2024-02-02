import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset


import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer

from datasets import DatasetDict, load_dataset
import matplotlib.pyplot as plt

INPUT_CHANNELS = 3
INPUT_HEIGHT = 32
INPUT_WIDTH = 32
PIXEL_MAX = 255
PIXEL_MIN = 0


class Encoder(nn.Module):
    def __init__(self, c_hid, latent_dim):
        super().__init__()

        act = nn.GELU

        conv = [
            # output channels, kernel_size, padding, stride
            (c_hid * 1, 3, 1, 2),
            (c_hid * 1, 3, 1, 1),
            (c_hid * 2, 3, 1, 2),
            (c_hid * 2, 3, 1, 1),
            (c_hid * 2, 3, 1, 2),
        ]

        conv_layers = []

        prev_channels = INPUT_CHANNELS
        for next_channels, kernel_size, padding, stride in conv:
            conv_layers.append(
                nn.Conv2d(
                    prev_channels,
                    next_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                )
            )
            conv_layers.append(act())
            prev_channels = next_channels

        self.net = nn.Sequential(
            *conv_layers, nn.Flatten(), nn.Linear(2 * 16 * c_hid, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, c_hid, latent_dim):
        super().__init__()

        act = nn.GELU

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * c_hid),
            act(),
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                2 * c_hid,
                2 * c_hid,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),  # 4x4 => 8x8
            act(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act(),
            nn.ConvTranspose2d(
                2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 8x8 => 16x16
            act(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act(),
            nn.ConvTranspose2d(
                c_hid,
                INPUT_CHANNELS,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),  # 16x16 => 32x32
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.size()[0], -1, 4, 4)
        x = self.net(x)
        return x


# sanity checks
def sanity_check():
    batch_size = 5
    c_hid = 6
    latent_dim = 64

    encoder = Encoder(c_hid, latent_dim)
    decoder = Decoder(c_hid, latent_dim)

    img = torch.zeros(batch_size, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    z = encoder(img)
    assert list(z.size()) == [batch_size, latent_dim]
    reconstructed = decoder(z)
    assert list(reconstructed.size()) == [
        batch_size,
        INPUT_CHANNELS,
        INPUT_HEIGHT,
        INPUT_WIDTH,
    ]


class AutoEncoder(LightningModule):
    def __init__(self, c_hid, latent_dim, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate

        self.encoder = Encoder(c_hid, latent_dim)
        self.decoder = Decoder(c_hid, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def x_x_hat_loss(self, batch):
        x = batch
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x, x_hat, reduction="none")
        loss = torch.sum(loss, dim=[1, 2, 3]).mean()
        return loss

    def common_step(self, batch, log_mode=None):
        loss = self.x_x_hat_loss(batch)
        if log_mode is not None:
            self.log_dict({f"{log_mode}_loss": loss.item()})
        return loss

    def training_step(self, batch):
        return self.common_step(batch)

    def validation_step(self, batch):
        return self.common_step(batch)

    def test_step(self, batch):
        return self.common_step(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
        )
        return [{"optimizer": optimizer, "scheduler": scheduler, "monitor": "val_loss"}]


class NormalizedDataset(Dataset):
    def __init__(self, split):
        dataset = load_dataset("cifar10").with_format("torch")
        assert type(dataset) == DatasetDict
        self.dataset = dataset[split]
        assert self.dataset is not None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset.__getitem__(idx)
        x = item["img"] * 1.0
        x = torch.permute(x, [2, 0, 1])
        x = (x - PIXEL_MIN) / (PIXEL_MAX - PIXEL_MIN)  # [0, 1]
        x = x * 2 - 1  # [-1, 1]
        return x


class MyDataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = NormalizedDataset("train")
        self.test_dataset = NormalizedDataset("test")

    def create_dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self.create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader(self.test_dataset)

    def test_dataloader(self):
        return self.create_dataloader(self.test_dataset)


class GenerateCallback(Callback):
    def __init__(self, imgs, every_n_epochs=1):
        super().__init__()
        self.imgs = imgs
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs == 0:
            training = module.training
            with torch.no_grad():
                original = self.imgs.detach().to(module.device)
                imgs = self.imgs.detach().to(module.device)
                imgs = module.forward(imgs)
            module.train(training)

            imgs = torch.cat([original, imgs])

            grid = (
                torchvision.utils.make_grid(
                    imgs, nrow=original.size()[0], normalize=True
                )
                .detach()
                .cpu()
            )

            plt.imshow(grid.permute(1, 2, 0))
            plt.savefig(f"./out/{epoch}.png")


def special_imgs():
    size = (1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    img1 = torch.randn(*size)

    img2 = torch.zeros(*size)
    img2[:, 0, :, :] = 0.8
    img2[:, 1, :, :] = 0.0
    img2[:, 2, :, :] = -0.3

    s = INPUT_HEIGHT // 2
    img3 = torch.zeros(*size)
    img3[:, :, :s, :s] = -0.8
    img3[:, :, :s, s:] = -0.4
    img3[:, :, s:, :s] = 0.4
    img3[:, :, s:, s:] = 0.8

    return torch.cat([img1, img2, img3], dim=0)


# sanity check:
# sanity_check()

# config
batch_size = 256
num_workers = 20
learning_rate = 1e-3
epochs = 30

c_hid = 16
latent_dim = 64

record_imgs = 4

# data
data = MyDataModule(batch_size, num_workers)

# prepare the images to record
imgs = next(iter(data.test_dataloader()))[:record_imgs, :, :, :]
imgs = torch.cat([special_imgs(), imgs], dim=0)

# create the network
module = AutoEncoder(c_hid, latent_dim, learning_rate)

# train
trainer = Trainer(
    min_epochs=1,
    max_epochs=epochs,
    callbacks=[GenerateCallback(imgs)],
)
trainer.fit(module, data)
trainer.validate(module, data)
trainer.test(module, data)
