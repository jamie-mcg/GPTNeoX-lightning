# LLM Template project

This repo is still under development but you can run a simple test by running the following command:

```
python3 main.py {fit,validate,etc.} --model=configs/models/pythia-14m.yaml --trainer=configs/trainer.yaml --data=configs/datasets/pile.yaml
```

Where the {subcommands} are the PyTorch CLI subcommands. Change the model architecture by going into `configs/models/pythia-14m.yaml` file.

Eventually, this will be an extensively documented and flexible codebase, which I'm hoping will be useful for anyone wishing to get started training and playing around with LLMs.

Documentation will be updated soon, along with a blog post to accompany this project... stay tuned!