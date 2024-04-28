# Introduction

- waiting

# Abstract

- Cloth-changing person re-identification (Re-ID) is an emerging research theme aimed at identifying individuals after clothing change. Many contemporary approaches focus on disentangling clothing features and solely employ unrelated to clothing for identification. However, the absence of ground truth poses a significant challenge to the disentanglement process, these methods may introduce unintended noise and degrade the overall performance. To mitigate this issue, we propose a novel framework, termed Attention-enhanced Controllable Disentanglement Re-ID (ACD-ReID). In CD-ReID, we design an Attention-enhancing Disentanglement Branch (ADB) where human parsing masks are introduced to guide the controllable disentanglement of clothing features and clothing-unrelated features. Moreover, we propose two novel attention mechanisms: Dynamic Interaction-Faraware Aggregation Attention (DI-FAA) and Dynamic Interaction-Positional Relevance Attention (DI-PRA), which are specifically designed to enhance the representation of unclothed body features and human contour features. Experimental results on LTCC, PRCC, VC-Clothes, DeepChange, and CCVID datasets demonstrate the superiority of our approach over representative state-of-the-art CC-ReID methods. For the cloth-changing setting, the mAP of our network on PRCC, LTCC, VC-Clothes, and DeepChange datasets are 58.5%, 19.5%, and 19.7%, and the Rank-1 are 58.6%, 41.5%, and 54.4%. In addition, the model also obtains 81.5% of mAP and 83.4% of Rank-1 on the video dataset CCVID.


# Dependencies

- Python 3.6

- PyTorch 1.6.0

- yacs

- apex



# Arithmetic Support

- The model is trained on two RTX 4090 GPUs

