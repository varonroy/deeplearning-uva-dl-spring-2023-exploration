import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision.datasets import CIFAR100
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule

import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv

import os
import math

# load env
load_dotenv()
TORCH_DIR = os.getenv("TORCH_DIR")
TORCH_HOME = os.getenv("TORCH_HOME")
CACHE_DIR = os.getenv("CACHE_DIR")

assert TORCH_DIR is not None, "TORCH_DIR must be set"
assert TORCH_HOME is not None, "TORCH_HOME must be set"
assert CACHE_DIR is not None, "CACHE_DIR must be set"

# configure gpu
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")

# load the base model
base_model = torchvision.models.resnet34(weights="IMAGENET1K_V1")
base_model.fc = nn.Identity()  # type: ignore
base_model.classifier = nn.Identity()  # type: ignore
base_model.eval()
for p in base_model.parameters():
    p.requires_grad_(False)


# extract features form the dataset and save them to a cache file
@torch.no_grad()
def extract_features(base_model, save_file, train):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = base_model.to(device)

    save_file = os.path.join(CACHE_DIR, save_file)
    if not os.path.isfile(save_file):
        print(f"save file `{save_file}` not found, generating...")
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.2),
            ]
        )
        dataset = CIFAR100(
            root=TORCH_DIR,
            train=train,
            transform=transform,
            download=True,
        )
        extracted = []
        loader = DataLoader(
            dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=10
        )
        for imgs, _ in loader:
            imgs = imgs.to(device)
            out = base_model(imgs).cpu()
            extracted.append(out)
        extracted = torch.cat(extracted, dim=0)
        torch.save(extracted, save_file)
        return extracted
    else:
        return torch.load(save_file)


train_data_features = extract_features(base_model, "train_data_features.tar", True)
test_data_features = extract_features(base_model, "test_data_features.tar", False)
print("loaded extracted features")
print(" - train features:", train_data_features.size())
print(" - test features :", test_data_features.size())
print()


# split a dataset while keeping the same number of labels pe
def train_validation_datasets(val_items_per_class):
    assert val_items_per_class > 0

    dataset = CIFAR100(
        root=TORCH_DIR,
        train=True,
        download=True,
    )

    labels = dataset.targets  # list of indices
    num_classes = len(set(labels))

    # convert the indices to a tensro
    labels = torch.LongTensor(labels)

    # the indices of the sorted labels
    indices = torch.range(start=0, end=labels.size()[0] - 1, step=1, dtype=torch.long)
    indices_val = []
    indices_train = []
    for c in range(num_classes):
        mask = labels == c  # mask all labels with the class
        c_indices = torch.masked_select(indices, mask)
        assert (
            c_indices.size()[0] >= val_items_per_class
        ), f"Not enough data in class {c}"

        c_indices_val = c_indices[:val_items_per_class]
        c_indices_train = c_indices[val_items_per_class:]

        indices_val.append(c_indices_val)
        indices_train.append(c_indices_train)

        # # debug
        # c_labels_val = torch.index_select(labels, dim=0, index=c_indices_val)
        # c_labels_train = torch.index_select(labels, dim=0, index=c_indices_train)
        # print(c_labels_val)
        # print(c_labels_train)
        # exit(0)

    return torch.stack(indices_train, dim=0), torch.stack(indices_val, dim=0)


print("preparing train / test indices")
indices_train, indices_val = train_validation_datasets(50)
print(" - train indices     :", indices_train.size())
print(" - validation indices:", indices_val.size())


# index test set
def index_test_set():
    dataset = CIFAR100(
        root=TORCH_DIR,
        train=False,
        download=True,
    )

    labels = dataset.targets  # list of indices
    num_classes = len(set(labels))
    labels = torch.LongTensor(labels)
    indices = torch.range(start=0, end=labels.size()[0] - 1, step=1, dtype=torch.long)

    test_indices = []
    for c in range(num_classes):
        mask = labels == c  # mask all labels with the class
        c_indices = torch.masked_select(indices, mask)
        test_indices.append(c_indices)

        # # debug
        # c_labels = torch.index_select(labels, dim=0, index=c_indices)
        # print(c_labels)
        # exit(0)

    return torch.stack(test_indices, dim=0)


print("preparing test indices")
indices_test = index_test_set()


class IndexedGropusDataset(Dataset):
    def __init__(self, features, indices, images_per_item, randomize_items):
        """
        params:
            `features` - a tensor of the extracted features. Tensor [num_items, feature_dim]
            `indices`  - the indices of the classes in the dataset. Tensor [num_classes, items_per_class].
        """
        super().__init__()
        self.features = features
        self.indices = indices
        self.randomize_items = randomize_items
        self.images_per_item = images_per_item
        self.num_classes, self.items_per_class = indices.size()
        assert images_per_item <= self.items_per_class

    def __len__(self):
        return self.features.size()[0]

    def __getitem__(self, idx):
        if self.randomize_items:
            import random
            from random import randint

            c1 = randint(0, self.num_classes - 1)
            c2 = randint(0, self.num_classes - 1)
            while c2 == c1:
                c2 = (c2 + 1) % self.num_classes
            l = list(range(self.items_per_class))
            random.shuffle(l)
            raw_indices = (
                l[: self.images_per_item],
                randint(0, self.items_per_class - 1),
            )
        else:
            c1 = idx % self.num_classes
            c2 = (idx * 7 + 1) % self.num_classes
            while c2 == c1:
                c2 = (c2 + 1) % self.num_classes
            i1 = idx % (self.items_per_class - self.images_per_item)
            i2 = (idx * 7 + 1) % self.items_per_class
            raw_indices = (
                list(range(self.items_per_class))[i1 : i1 + self.images_per_item - 1],
                i2,
            )

        c1_indices = self.indices[c1][raw_indices[0]]
        c2_index = self.indices[c2][raw_indices[1]]
        indices = torch.cat([c1_indices, c2_index.unsqueeze(dim=-1)], dim=0)
        indices = indices[torch.randperm(indices.size()[0])].to(dtype=torch.long)
        anomaly_index = torch.argmax((indices == c2_index) * 1.0).to(dtype=torch.long)
        features = self.features[indices]
        return features, indices, anomaly_index


class AnomalyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_features,
        test_data_features,
        indices_train,
        indices_val,
        indices_test,
        images_per_item,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.train_dataset = IndexedGropusDataset(
            train_data_features, indices_train, images_per_item, True
        )
        self.val_dataset = IndexedGropusDataset(
            train_data_features, indices_val, images_per_item, False
        )
        self.test_dataset = IndexedGropusDataset(
            test_data_features, indices_test, images_per_item, False
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

    def dataloader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)  # type: ignore

    def train_dataloader(self):
        return self.dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.dataloader(self.val_dataset)

    def test_dataloader(self):
        return self.dataloader(self.test_dataset)

    def debug_show(self):
        assert TORCH_DIR is not None

        dataset = CIFAR100(
            root=TORCH_DIR,
            train=True,
            download=True,
        )

        _, indices, anomaly = next(iter(self.train_dataloader()))
        indices = indices.tolist()[0]
        anomaly = anomaly.item()
        imgs = [dataset.__getitem__(i)[0] for i in indices]
        _, axes = plt.subplots(
            1, len(imgs), figsize=(15, 5)
        )  # Adjust figsize as needed

        for i, (ax, img) in enumerate(zip(axes, imgs)):
            if i == anomaly:
                ax.set_title("anomaly")
            ax.imshow(img)
            ax.axis("off")

        plt.show()
        exit(0)

class EncoderModuleLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

        self.linear_net = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)

        linear_out = self.linear_net(x)
        x = x + linear_out
        x = self.norm2(x)

        return x


class EncoderModule(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([EncoderModuleLayer(d_model, num_heads=num_heads) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Model(nn.Module):
    def __init__(self, input_dim, model_dim, out_dim, num_heads, num_layers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
            nn.LayerNorm(model_dim),
            # nn.TransformerEncoder(
            #     nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True),
            #     num_layers=num_layers,
            # ),
            EncoderModule(d_model=model_dim, num_heads=num_heads, num_layers=num_layers),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, out_dim),
            # nn.Linear(input_dim, model_dim),
            # nn.ReLU(),
            # nn.LayerNorm(model_dim),
            # nn.TransformerEncoder(
            #     nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads),
            #     num_layers=num_layers,
            # ),
            # nn.Linear(model_dim, model_dim),
            # nn.LayerNorm(model_dim),
            # nn.ReLU(),
            # nn.Linear(model_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# define the detector module - it is reponsible for using the base model,
# training the appended transformer and performing the classification.
class Detector(pl.LightningModule):
    def __init__(self, model, lr, num_classes):
        super().__init__()
        self.model = model
        self.lr = lr
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, log_mode=None):
        features, _, anomaly_idx = batch
        # label = nn.functional.one_hot(anomaly_idx, self.num_classes)

        #  out: [batch, seq_len, 1] -> [batch_size, seq_len]
        out = self.forward(features).squeeze(-1)

        # run cross entropy loss
        loss = nn.functional.cross_entropy(
            out,
            anomaly_idx,
        )
        # print(out)
        # print(anomaly_idx)
        # print(loss)
        # exit(0)

        # TODO: use accuracy Metric
        acc = (out.argmax(-1) == anomaly_idx).float().mean()

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
            }
        ]

    def debug_permutation(self, data: AnomalyDataModule):
        with torch.no_grad():
            loader = data.train_dataloader()

            features, _, _ = next(iter(loader))
            out = self.forward(features).squeeze(dim=-1)

            indices = np.random.permutation(features.size()[1])
            features_permuted = features[:, indices, :]
            out_permuted = self.forward(features_permuted).squeeze(dim=-1)

            print('indices')
            print(indices)

            print('------')
            print(features)
            print(features_permuted)
            print('diff:', (features[:, indices, :] - features_permuted).abs().max())
            print('------')

            print('------')
            print(out)
            print(out[:, indices])
            print(out_permuted)
            print('diff:', (out[:, indices] - out_permuted).abs().max().item())
            print('------')

            exit(0)


INPUT_DIM = 512
NUM_CLASSES = 100

# config
images_per_item = 10
learning_rate = 1e-4

epochs = 10
batch_size = 128
num_workers = 13

model_dim = 512
num_heads = 8
num_layers = 8

# create the data module
data = AnomalyDataModule(
    train_data_features=train_data_features,
    test_data_features=test_data_features,
    indices_train=indices_train,
    indices_val=indices_val,
    indices_test=indices_test,
    images_per_item=images_per_item,
    batch_size=batch_size,
    num_workers=num_workers,
)

# data.debug_show()

# create the model
model = Model(
    input_dim=INPUT_DIM,
    model_dim=model_dim,
    out_dim=1,
    num_heads=num_heads,
    num_layers=num_layers,
)
detector = Detector(model, lr=learning_rate, num_classes=NUM_CLASSES)

# detector.debug_permutation(data)

# train
trainer = pl.Trainer(min_epochs=1, max_epochs=epochs)
trainer.fit(detector, data)
trainer.validate(detector, data)
trainer.test(detector, data)

