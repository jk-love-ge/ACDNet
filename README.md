# Introduction

- waiting

# Abstract

- Download the pre-processed datasets that we used from the link (password: dg1a) and unzip them to ./datasets folders.


# Dependencies

- Python 3.6

- PyTorch 1.6.0

- yacs

- apex

# How to use：
- Path: Replace _C.DATA.ROOT and _C.OUTPUT in configs/default_img.py&default_vid.pywith your own data path and output path, respectively.

- Training: For dataset_name dataset: python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset dataset_name --cfg configs/res50_cels_cal.yaml --gpu 0,1 --spr 0 --sacr 0.05 --rr 1.0



# Arithmetic Support

- The model is trained on two RTX 4090 GPUs

