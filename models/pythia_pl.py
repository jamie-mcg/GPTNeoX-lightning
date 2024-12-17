import torch
from torch import nn
import yaml
import os
from argparse import Namespace
from dataclasses import asdict

import lightning.pytorch as pl
import torchmetrics

from models.gpt_neox import GPTNeoXForCausalLM

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from utils import TrainingOptions, InferenceOptions

MODEL_REGISTRY_PATH = os.environ.get("MODEL_REGISTRY_PATH")


class PythiaModel(nn.Module):
    def __init__(self, model, pretrained, model_name_or_path, tokenizer_name_or_path):
        super().__init__()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(MODEL_REGISTRY_PATH, tokenizer_name_or_path)
        )

        # Optionally load pretrained weights
        if pretrained:
            pretrained_model = GPTNeoXForCausalLM.from_pretrained(
                os.path.join(MODEL_REGISTRY_PATH, model_name_or_path)
            )
            self.model.gpt_neox.load_state_dict(pretrained_model.state_dict())
        self.model.train()  # Set the model to train mode

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class LLM(pl.LightningModule):
    def __init__(
        self,
        model: PythiaModel,
        training: TrainingOptions,
        inference: InferenceOptions,
    ):
        super().__init__()
        # TODO: Why do I need to do this?
        hps = Namespace(
            **{
                "training": Namespace(**asdict(training)),
                "inference": Namespace(**asdict(inference)),
            }
        )
        self.save_hyperparameters(hps)

        self.model = model

        self.train_metrics = (
            torchmetrics.MetricCollection(self.hparams.training.metrics)
            if self.hparams.training.metrics
            else torchmetrics.MetricCollection({})
        )
        self.infer_metrics = (
            torchmetrics.MetricCollection(self.hparams.inference.metrics)
            if self.hparams.inference.metrics
            else torchmetrics.MetricCollection({})
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]
        self.log(
            "train/loss",
            loss,
            sync_dist=True,
            on_step=self.training,
            on_epoch=not self.training,
        )
        self.train_metrics.update(outputs.logits, batch["input_ids"])
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs["loss"]
        self.log(
            "val/loss",
            val_loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.infer_metrics.update(outputs.logits, batch["input_ids"])
        return val_loss

    def on_validation_epoch_end(self):
        train_metrics, infer_metrics = (
            self.train_metrics.compute(),
            self.infer_metrics.compute(),
        )

        for key, value in train_metrics.items():
            self.log(f"train/{key}", value, on_epoch=True, sync_dist=True)

        for key, value in infer_metrics.items():
            self.log(f"val/{key}", value, on_epoch=True, sync_dist=True)

        self.train_metrics.reset(), self.infer_metrics.reset()

    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self(input_ids, attention_mask=attention_mask)
        return outputs["logits"]

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.training.optimizer)(
            self.parameters(), **self.hparams.training.optimizer_args
        )

        if self.hparams.training.lr_scheduler:
            scheduler = getattr(
                torch.optim.lr_scheduler, self.hparams.training.lr_scheduler
            )(optimizer, **self.hparams.training.lr_scheduler_args)

            return (
                [optimizer],
                [
                    {
                        "scheduler": scheduler,
                        **self.hparams.training.pl_lr_scheduler_args,
                    }
                ],
            )
        else:
            return optimizer

    # def freeze_layers(self, layers_to_freeze):
    #     for name, param in self.model.named_parameters():
    #         if any(layer in name for layer in layers_to_freeze):
    #             param.requires_grad = False
    #             print(f"Froze layer: {name}")
