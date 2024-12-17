import torch
from lightning.pytorch.cli import LightningCLI

from models import PythiaModel, LLM
from data import PileDataModule

torch.set_float32_matmul_precision("medium")


def main():
    cli = LightningCLI(
        # LLM,
        # PileDataModule,
        # run=True,
        # subclass_mode_model=True
        parser_kwargs={
            "parser_mode": "omegaconf"
        }
    )


if __name__ == "__main__":
    main()
