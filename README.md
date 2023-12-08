# Introduction

- This article was submitted for publication in “Intelligent Data Analysis”. The authors are Yiyuan Ge, M.S. and Mingxin Yu, Ph.

# Abstract

- The key to solving the cloth-changing person re-identification (Re-ID) problem lies in separating discriminative information that is independent of clothing, such as pose, silhouette and face. Most current approaches disentangle clothes-unrelated features by modeling clothes-related features. However, due to the lack of ground truth for guidance, this disentanglement process remains uncontrolled. These methods may unintentionally introduce noise into the system, thereby impairing the performance of cloth-changing person re-identification. To mitigate this issue, we propose a novel framework, termed Controllable Disentanglement Re-ID (CD-ReID), which employs reconstruction learning and attention enhancement for controllable separation of clothing-related and unrelated features. Specifically, our framework introduces an Attention-Enhancing Disentanglement Branch (ADB), designed to reconstruct clothing-related features, clothing-unrelated features, and contour features. Furthermore, we propose two novel attention mechanisms: Dynamic Interaction-Faraware Aggregation Attention (DI-FAA) and Dynamic Interaction-Positional Relevance Attention (DI-PRA). These mechanisms aim to augment the expressive capacity of the disentangled features. Experiments on LTCC, PRCC and CCVID demonstrate the superiority of our approach over representative state-of-the-art CC-ReID methods. For the cloth-changing setting, the mAP of the network on PRCC/LTCC dataset is 58.5%/19.5%, and the Rank-1 is 58.6%/41.5%. In addition, the model also obtain 81.5% of mAP and 83.4% of Rank-1 on the video dataset CCVID under cloth-changing setting


# Dependencies

- Python 3.6

- PyTorch 1.6.0

- yacs

- apex


# Data Preparation

- Download the pre-processed datasets that we used from the [link](https://pan.baidu.com/s/1LwAyB1R86P3xMZxIPm1vwQ) (password: dg1a) and unzip them to ./datasets folders.

# Arithmetic Support

- The model is trained on a single RTX 4090 GPU

