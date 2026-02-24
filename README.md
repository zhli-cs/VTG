# Versatile Transferable Unlearnable Example Generator

Code for NeurIPS 2025 Paper ["Versatile Transferable Unlearnable Example Generator"](https://openreview.net/forum?id=pe8Dr5jtjV) by Zhihao Li*, Jiale Cai*, Gezheng Xu, Hao Zheng, Qiuyue Li, Fan Zhou, Shichun Yang, Charles Ling, Boyu Wang.

## Abstract
The rapid growth of publicly available data has fueled deep learning advancements but also raises concerns about unauthorized data usage. Unlearnable Examples (UEs) have emerged as a data protection strategy that introduces imperceptible perturbations to prevent unauthorized learning. However, most existing UE methods produce perturbations strongly tied to specific training sets, leading to a significant drop in unlearnability when applied to unseen data or tasks. In this paper, we argue that for broad applicability, UEs should maintain their effectiveness across diverse application scenarios. To this end, we conduct the first comprehensive study on the transferability of UEs across diverse and practical yet demanding settings. Specifically, we identify key scenarios that pose significant challenges for existing UE methods, including varying styles, out-of-distribution classes, resolutions, and architectures. Moreover, we propose **Versatile Transferable Generator** (VTG), a transferable generator designed to safeguard data across various conditions. Specifically, VTG integrates Adversarial Domain Augmentation (ADA) into the generator’s training process to synthesize out-of-distribution samples, thereby improving its generalizability to unseen scenarios. Furthermore, we propose a Perturbation-Label Coupling (PLC) mechanism that leverages contrastive learning to directly align perturbations with class labels. This approach reduces the generator’s reliance on data semantics, allowing VTG to produce unlearnable perturbations in a distribution-agnostic manner. Extensive experiments demonstrate the effectiveness and broad applicability of our approach.

## Requirements
- Setup a conda environment and install some prerequisite packages.
```
conda create -n your_env_name python=3.13
conda activate your_env_name
pip install -r requirements.txt
```

## Running Experiments
We provide VTG generator checkpoints pretrained on CIFAR-10, CIFAR-100, and SVHN. You can also train your own model using the following scripts.

### Training
- Train VTG on CIFAR-10:
```
bash train_cifar10.sh
```

### Evaluation
- Evaluate VTG on CIFAR-10:
```
bash test_cifar10.sh
```

- Evaluate VTG's transferability
```
bash test_cifar100.sh / bash test_svhn.sh
```

## Acknowledgement
This codebase is partially based on [EMN](https://github.com/HanxunH/Unlearnable-Examples).


## Citation
If you find our paper/code useful, please cite the following paper:
```
@inproceedings{
li2025versatile,
title={Versatile Transferable Unlearnable Example Generator},
author={Zhihao Li and Jiale Cai and Gezheng Xu and Hao Zheng and Qiuyue Li and Fan Zhou and Shichun Yang and Charles Ling and Boyu Wang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025}
}
``` 
