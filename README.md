# Harmonization 2.0

## Environment Setup

```
conda create -n adv python=3.9 -y
conda activate hmn
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install timm==0.9.0 torchmetrics==1.0.1
pip install wandb accelerate pathlib numpy tqdm scipy
```
