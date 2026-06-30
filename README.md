# MiRA: Reweighting Framewise Attention in Video Transformers for Facial Expression Understanding.
[Reweighting Framewise Attention in Video Transformers for Facial Expression Understanding (ECCV 2026)](https://arxiv.org/abs/2606.30611) <br>
Seongro Yoon<sup>1</sup> &nbsp;&nbsp;, 
Donghyeon Cho<sup>2</sup> &nbsp;&nbsp;, 
Jinsun Park<sup>3</sup> &nbsp;&nbsp;,
François Brémond<sup>1</sup> <br>
<sup>1</sup> Inria, France &nbsp;&nbsp;
<sup>2</sup> Hanyang University, South Korea &nbsp;&nbsp; 
<sup>3</sup> Pusan National University, South Korea

<p align="center">
  <img src="assets/fig1_frame_marginal_modules.png" width="70%">
</p>
<p align="center">
  <img src="assets/fig2_illustration_method.png" width="70%">
</p>

MiRA (Marginal-induced Attention Redistribution) is a lightweight plug-in module for foundational video transformers that introduces frame-marginal attention reweighting for facial emotion understanding. It encourages more complementary spatio-temporal facial representations by redistributing attention across frames, consistently improving performance with minimal additional computation. MiRA supports **Exact mode** for principled post-softmax attention redistribution and **FlashLite mode** for efficient FlashAttention-compatible approximation.

## Datasets

**Pre-training: million-scale unlabeled facial videos**
- [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) (1.2M videos)

**Fine-tuning: downstream facial expression recognition**
- [DFEW](https://dfew-dataset.github.io/) (12K video clips)
- [MAFW](https://mafw-database.github.io/MAFW/) (10K video clips)
- [FERV39k](https://wangyanckxx.github.io/Proj_CVPR2022_FERV39k.html) (39K video clips)

**k-NN Probing: micro-expression recognition**
- [SAMM](https://repository.mmu.ac.uk/articles/journal_contribution/SAMM_A_Spontaneous_Micro-Facial_Movement_Dataset/32439684?file=64997748)
- [MMEW](https://github.com/benxianyeteam/MMEW-Dataset)

## Pre-training

## Fine-tuning with pre-trained models

