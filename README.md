# How to train your LLM (with Lightning)

This repo is meant as a template for anyone wishing to experiment with LLMs on a more fundamental level. It is designed to piggy-back off all the great features of PyTorch Lightning and Lightning CLI - which allows basically full control of your experiments via config files!

## Usage instructions

To set things up, you can use the usual `pip install -r requirements.txt` command, or indeed, create a docker image/container using the dockerfile.

Once you have this, you can run the following basic command to run a quick test with IMDB movie review data and a small Pythia model.

```
python3 main.py {fit,validate,etc.} --model=configs/models/pythia-14m.yaml --trainer=configs/trainer.yaml --data=configs/datasets/pile.yaml
```

Where the {subcommands} are the PyTorch CLI subcommands. Change the model architecture by going into `configs/models/pythia-14m.yaml` file - or defining your own using the GPTNeoX API.

## Disclaimer

Currently, this repo is still in its infancy. In the near future, I'm hoping this will be an extensively documented and flexible codebase which streaming dataset examples and extra tools for analysing your model/data and pipelines (i.e. via callbacks). Either way, I'm hoping this will be useful for anyone wishing to get started training and playing around with LLMs.

Documentation will be updated as and when new features are added, along with a blog post to accompany this project... stay tuned!
