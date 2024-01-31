import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

import math


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout=0.0) -> None:
        super().__init__()
        print(" - ", input_dim)
        self.attn = MultiheadAttention(input_dim, num_heads, batch_first=True)

        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self, num_blocks, input_dim, num_heads, hidden_dim, dropout=0.0
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(input_dim, num_heads, hidden_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask=mask)
        return x


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class TransformerPredictor(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_classes,
        num_heads,
        num_blocks,
        dropout,
        positional_encoding=True,
        lr=1e-3,
        warmup=100,
        max_iters=1000,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.warmup = warmup
        self.max_iters = max_iters

        self.input = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, model_dim),
            PositionalEncoding(model_dim) if positional_encoding else nn.Identity(),
        )
        self.encoder = Encoder(
            num_blocks=num_blocks,
            input_dim=model_dim,
            num_heads=num_heads,
            hidden_dim=model_dim,
            dropout=dropout,
        )
        self.output = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        )

    def forward(self, x, mask=None):
        """
        input:
            x - Tensor of shape [batch, seq, input_dim]
        """
        x = self.input(x)
        x = self.encoder(x, mask)
        x = self.output(x)
        return x

    def common_step(self, batch, mask=None, log_mode=None):
        """
        input:
            batch: tuple of (x, label) both are tensor of size (batch_size, seq_len)
        """
        x = batch[0]
        label = batch[1]

        # convert input to a one-hot vector: [batch, seq_len, num_classes]
        x = F.one_hot(x, num_classes=self.num_classes).float()

        # run the network
        out = self.forward(x)

        # run cross entropy loss
        #  out:   [batch, seq_len, num_classes] -> [batch_size * seq_len, num_classes]
        #  label: [batch, seq_len]              -> [batch_size * seq_len]
        loss = F.cross_entropy(
            out.view(-1, self.num_classes),
            label.view(-1),
        )

        # calculate the accuracy
        acc = (out.argmax(dim=-1) == label).float().mean()
        # TODO: use log Metric
        if log_mode is not None:
            self.log_dict(
                {
                    f"{log_mode}-loss": loss.item(),
                    f"{log_mode}-acc": acc.item(),
                }
            )

        return loss

    def training_step(self, batch):
        return self.common_step(batch, log_mode="train")

    def validation_step(self, batch):
        return self.common_step(batch, log_mode="validation")

    def test_step(self, batch):
        return self.common_step(batch, log_mode="test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return [
            {
                "optimizer": optimizer,
                # TODO: figure out how to properly set the scheduler
                # "scheduler": CosineWarmupScheduler(
                #     optimizer,
                #     warmup=self.warmup,
                #     max_iters=self.max_iters,
                # ),
                # "interval": "step",
            }
        ]


class RevDataset(Dataset):
    """
    example item:
        (tensor([9, 6, 2, 0, 6, 2, 7, 9, 7, 3, 3, 4, 3, 7, 0, 9]),
         tensor([9, 0, 7, 3, 4, 3, 3, 7, 9, 7, 2, 6, 0, 2, 6, 9]))
    """

    def __init__(self, num_classes: int, seq_len: int, dataset_size: int):
        super().__init__()
        self.data = torch.randint(low=0, high=num_classes, size=(dataset_size, seq_len))  # type: ignore

    def __len__(self):
        return self.data.size()[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        label = torch.flip(x, dims=(0,))
        return (x, label)


class RevDataMoudule(pl.LightningDataModule):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        dataset_size: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.dataset_size = (dataset_size,)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_data = RevDataset(num_classes, seq_len, dataset_size)
        self.validation_data = RevDataset(num_classes, seq_len, dataset_size // 10)
        self.test_data = RevDataset(num_classes, seq_len, dataset_size // 10)

    def dataloader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)  # type: ignore

    def train_dataloader(self):
        return self.dataloader(self.train_data, shuffle=True)

    def val_dataloader(self):
        return self.dataloader(self.validation_data)

    def test_dataloader(self):
        return self.dataloader(self.test_dataloader)


# prefer performance
torch.set_float32_matmul_precision('medium')

# config
num_classes = 10
seq_len = 10
dataset_size = 1024 * 8
batch_size = 256
num_workers = 1

epochs = 10

model_dim = 32
num_heads = 1
num_blocks = 1
dropout = 0.0
lr = 2e-3
warmup = 50
max_iters = epochs * (dataset_size // batch_size + 1)

# create the data module
data = RevDataMoudule(
    num_classes=num_classes,
    seq_len=seq_len,
    dataset_size=dataset_size,
    batch_size=batch_size,
    num_workers=num_workers,
)

# create the model
model = TransformerPredictor(
    input_dim=num_classes,
    model_dim=model_dim,
    num_classes=num_classes,
    num_heads=num_heads,
    num_blocks=num_blocks,
    dropout=dropout,
    positional_encoding=True,
    lr=lr,
    warmup=warmup,
    max_iters=max_iters,
)

# train
trainer = pl.Trainer(min_epochs=1, max_epochs=epochs)
trainer.fit(model, data)
trainer.validate(model, data)

# example
print('example')
batch = next(iter(data.train_dataloader()))
x = batch[0][:3, :]
label = batch[1][:3, :]
out = model(F.one_hot(x, num_classes=num_classes).float()).argmax(-1)
print(x)
print(out)
print(label)

