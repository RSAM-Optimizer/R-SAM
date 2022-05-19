
# Random Sharpness-Aware Minimization (R-SAM) 




## Introduction

Currently, Sharpness-Aware Minimization (SAM) is proposed to seek the parameters that lie in a flat region to solve the generalization issue. For the sake of simplicity, SAM applies one-step gradient ascent for approximation to achieve the inner maximization. However, models are usually observed to locate in a sharp region, where the gradient is unstable, leading to poor performance on one-step gradient ascent. Based on this observation, we propose a novel random smoothing based SAM (R-SAM) algorithm. To be specific, R-SAM essentially smooths the loss landscape, based on which we are able to apply the one-step gradient ascent on the smooth weights for a much more accurate measurement of the inner maximization. 

## Installation

Install python dependencies by running (`Python>=3.6`):
```
pip install -r vit_jax/requirements.txt
```


## How to train ViT from scratch 

You can hit the command to train ViT-B-16 from scratch on ImageNet:  

```
bash run.sh
```

Before running this command, you may also need to provide the dataset path in `vit_jax/flags.py`. 

Noted that we use TPU v3-256 chips for ViT-B-16 training, the batch size in `run.sh` is 128 to make sure the global batch size is 4096 (128*(256/8)). 


Note: This repository was based on official implementation of Vision Transformer: 
[google-research/vision_transformer](https://github.com/google-research/vision_transformer).