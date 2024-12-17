import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

import torch
from torch import nn
import yaml

from models.gpt_neox import GPTNeoXForCausalLM, GPTNeoXConfig
from utils import *

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

torch.autograd.set_detect_anomaly(True)


def load_model_from_config(config_path):
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    model_config = GPTNeoXConfig(**config_data)
    layers_config = config_data["layers"]
    return GPTNeoXForCausalLM(model_config, layers_config)


class PythiaModel(pl.LightningModule):

    # TODO: Add paths into config for tokenizer and model separately
    def __init__(self, config_path, model_name_or_path="pythia-14m", pretrained=False):
        super().__init__()
        self.model = load_model_from_config(config_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/proj/mtklmadm/models/pythia_all/" + model_name_or_path
        )

        # Optionally load pretrained weights
        if pretrained:
            pretrained_model = GPTNeoXForCausalLM.from_pretrained(model_name_or_path)
            self.model.gpt_neox.load_state_dict(pretrained_model.state_dict())
        self.model.train()  # Set the model to train mode

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs["loss"]
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self(input_ids, attention_mask=attention_mask)
        return outputs["logits"]

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)

    # def freeze_layers(self, layers_to_freeze):
    #     for name, param in self.model.named_parameters():
    #         if any(layer in name for layer in layers_to_freeze):
    #             param.requires_grad = False
    #             print(f"Froze layer: {name}")


class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }


from lightning.pytorch.cli import LightningCLI


torch.set_float32_matmul_precision("medium")


def main():
    cli = LightningCLI(
        run=True,
        # model_class=PythiaModel,
        # datamodule_class=IMDbDataModule,
    )


if __name__ == "__main__":
    main()
