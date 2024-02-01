import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningDataModule, LightningModule

from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader, Dataset

INPUT_WIDTH = 28
INPUT_HEIGHT = 28
INPUT_CHANNELS = 1

MEAN = 120.0
STDDEV = 200.0


class Swish(nn.Module):
    def forward(self, x):
        return x * nn.functional.sigmoid(x)


class Model(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, hidden_size, kernel_size=5, stride=2, padding=4),
            Swish(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Flatten(),
            nn.Linear(hidden_size * 2 * 2, hidden_size),
            Swish(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        """
        input:  [batch, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH]
        output: [batch]
        """
        return self.net(x).squeeze()


class NormalizedDataset(Dataset):
    def __init__(self, split):
        dataset = load_dataset("mnist").with_format("torch")
        assert type(dataset) == DatasetDict
        self.dataset = dataset[split]
        assert self.dataset is not None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset.__getitem__(idx)
        x = item["image"] * 1.0
        x = x.unsqueeze(dim=0)
        return (x - MEAN) / STDDEV


class MyDataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.dataset = NormalizedDataset("train")
        self.batch_size = batch_size
        self.num_workers = num_workers

    def create_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def train_dataloader(self):
        return self.create_dataloader(self.dataset)


class Sampler:
    def __init__(
        self,
        model,
        sample_size,
        regen_size,
        step_size,
        num_steps,
    ):
        super().__init__()
        self.model = model
        self.sample_size = sample_size
        self.regen_size = regen_size
        self.step_size = step_size
        self.num_steps = num_steps

        self.buffer = torch.randn(
            sample_size, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH
        ).to("cuda")

    def sample(self):
        # create the indices of the images to replace
        from random import shuffle

        indices = list(range(self.sample_size))
        shuffle(indices)
        indices = indices[: self.regen_size]

        # detach from the buffer
        buffer = self.buffer.detach()

        # fill the indices with random noise
        buffer[indices] = torch.randn(
            self.regen_size, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH
        ).to(buffer.device)
        buffer.requires_grad_(True)
        buffer.retain_grad()

        # enable gradient calculation and record pytorch's state
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # disable model gradients
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # sampling loop
        for _ in range(self.num_steps):
            # reset the gradients
            if buffer.grad is not None:
                buffer.grad.detach_()
                buffer.grad.zero_()

            # add noise
            buffer.data.add_(0.005 * torch.randn_like(buffer))

            # calculate the gradietns
            (self.model(buffer)).sum().backward()

            assert buffer.grad is not None

            # apply gradient clipping for stabilization
            buffer.grad.data.clamp_(-0.03, 0.03)
            buffer.data.add_(self.step_size * buffer.grad)

        # return the model to its original state
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # restore pytorch's state
        torch.set_grad_enabled(had_gradients_enabled)

        # record the buffer
        self.buffer = buffer.detach()

        # return a copy of the buffer
        return buffer.detach()


class Generator(LightningModule):
    def __init__(self, model, learning_rate, sampler) -> None:
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.sampler = sampler

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.99)
        return [{"optimizer": optimizer, "scheduler": scheduler}]

    def training_step(self, real_images):
        real_images += torch.randn_like(real_images) * 0.005
        fake_images = self.sampler.sample()

        real_out = self.model(real_images)
        fake_out = self.model(fake_images)

        l_cd = fake_out.mean() - real_out.mean()
        l_rg = (fake_out**2).mean() + (real_out**2).mean()
        l = l_cd + 0.01 * l_rg

        self.log_dict(
            {
                "training-l-cd": l_cd.item(),
                "training-l-rg": l_cd.item(),
                "training-l": l.item(),
            },
        )

        return l


class GenerateCallback(Callback):
    def __init__(self, batch_size=8, num_steps=256, every_n_epochs=5):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs == 0:
            torch.set_grad_enabled(True)
            sampler = Sampler(module.model, 8, 0, 1e-3, 256)
            imgs = sampler.sample()
            torch.set_grad_enabled(False)

            imgs = imgs * STDDEV + MEAN
            imgs = imgs.cpu()

            from torchvision.utils import make_grid

            imgs = make_grid(imgs, normalize=True)
            plt.imshow(imgs.transpose(0, 2).transpose(0, 1))
            plt.savefig(f"./out/{epoch}.png")


# config
epochs = 51

num_workers = 10
batch_size = 256

sample_size = batch_size  # the size of the sampling buffer
regen_size = 4  # number of items in the sampling buffer to regenrate
step_size = 1e-3  # the step size when generating model samples
num_steps = 60  # the number of sampling steps to take

learning_rate = 1e-4
gradient_clip_val = 0.1

# data
data = MyDataModule(
    batch_size=batch_size,
    num_workers=num_workers,
)

# model and lightning module
model = Model()
sampler = Sampler(
    model=model,
    sample_size=sample_size,
    regen_size=regen_size,
    step_size=step_size,
    num_steps=num_steps,
)

generator = Generator(
    model=model,
    learning_rate=learning_rate,
    sampler=sampler,
)

# trainer
trainer = pl.Trainer(
    min_epochs=1,
    max_epochs=epochs,
    gradient_clip_val=gradient_clip_val,
    callbacks=[GenerateCallback()],
)
trainer.fit(generator, data)
