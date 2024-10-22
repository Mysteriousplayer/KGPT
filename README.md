# LORE
Enhancing Pre-trained ViTs for Downstream Task Adaptation: A Locality-Aware Prompt Learning Method[ACM MM 24]

Paper link: [https://openreview.net/forum?id=x7NIbrZ42w&noteId=x7NIbrZ42w]


## Abstract
> Vision Transformers (ViTs) excel in extracting global information from image patches. However, their inherent limitation lies in effectively extracting information within local regions, hindering their applicability and performance. Particularly, fully supervised pre-trained ViTs,such as VanillaViT and CLIP, face thechallenge of locality vanishing when adapting to downstream tasks. To address this, we introduce a novel LOcality-aware pRompt lEarning (LORE) method, aiming to improve the adaptation of pre-trained ViTs to downstreamtasks. LORE integrates a data-driven Black Box module (i.e., a pre-trained ViT encoder) with a knowledge-driven White Box module. The White Box module is a locality-aware prompt learning
mechanism to compensate for ViTs’ deficiency in incorporating local information. More specifically, it begins with the design of a Locality Interaction Network (LIN), which treats an image as a neighbor graph and employs graph convolution operations to enhance local relationships among image patches. Subsequently, a Knowledge-Locality Attention (KLA) mechanism is proposed to capture critical local regions from images, learning Knowledge-Locality (K-L) prototypes utilizing relevant semantic knowledge. Afterwards, K-L prototypes guide the training of a Prompt Generator (PG) to generate locality-aware prompts for images. The locality-aware prompts, aggregating crucial local information, serve as additional input for our Black Box module. Combining pre-trained ViTs with our locality-aware prompt learning mechanism, our Black-White Box model enables the capture of both global and local information, facilitating effective downstream task adaptation. Experimental evaluations across four downstream tasks demonstrate the effectiveness and superiority of our LORE. 

## Framework

![image](https://github.com/Mysteriousplayer/KGPT/blob/main/model_v6.png)

-We propose a novel LOcality-aware pRompt lEarning method (LORE) consisting of a data-driven Black Box module and a knowledge-driven White Box module for downstream task adaptation.

-To mitigate the problem of locality vanishing in pre-trained ViT models, we design a locality-aware prompt learning mechanism as our White Box module to compensate for the limited local information incorporating capacity of pre-trained ViTs.

-We develop a Knowledge-Locality Attention (KLA) mechanism to capture critical local regions from images.KLA learns K-L prototypes of images utilizing a semantic knowledge-locality matching strategy, which are then leveraged to optimize the training of our Prompt Generator (PG). 

-Experimental results on 4 kinds of downstream tasks, including 16 benchmark datasets, demonstrate the superiority of the proposed LORE method.

## Datasets
CIFAR-100, Imagenet-100, Tiny-Imagenet, and  Imagenet-1000. 

## Installation
Install all requirements required to run the code on a Python 3.x by:
> First, activate a new conda environment.
> 
> pip install -r requirements.txt

## Training
All commands should be run under the project root directory. 

```
sh run.sh

```
## Results
Results will be saved in log/.  

## Citation
If you found our work useful for your research, please cite our work:
```
@inproceedings{lore,
author = {Wang, Shaokun and Yu, Yifan and He, Yuhang and Gong, Yihong},
title = {Enhancing Pre-trained ViTs for Downstream Task Adaptation: A Locality-Aware Prompt Learning Method},
year = {2024},
booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
pages = {},
numpages = {}
}
```
We thank the following repo providing helpful functions in our work. 

xxxxxxx
