import os
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from mnist_model import MNISTModel

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

# Init our model
mnist_model = MNISTModel(PATH_DATASETS, BATCH_SIZE)

# Init DataLoader from MNIST Dataset
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

# Initialize a trainer
trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=5,
    progress_bar_refresh_rate=20,
)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)

trainer.test()

# # save the model
# trainer.save_checkpoint('./mnist_result.ckpt')
