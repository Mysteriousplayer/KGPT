# LORE
Enhancing Pre-trained ViTs for Downstream Task Adaptation: A Locality-Aware Prompt Learning Method [ACM MM 24]

Paper link: [https://openreview.net/forum?id=x7NIbrZ42w&noteId=x7NIbrZ42w]

A poster of the paper has been uploaded.

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
For the classification task, we conducted experiments on three kinds of datasets: (1) Natural datasets: CIFAR-10, CIFAR-100, DTD, and ImageNet. (2) Fine-grained datasets: Flowers102, Stanford-Cars, FGVCAircraft, and StanfordDogs. (3) Specialized datasets: EuroSAT, Resisc45, UCF101, and Pattern. 

The comprehensive statistics of the classification datasets are presented in this Table. For StanfordDogs, CIFAR-10, CIFAR-100, DTD, ImageNet, Resisc45, and Pattern, we followed the official dataset split strategy. For Flowers102, StanfordCars, Aircraft, EuroSAT, and UCF, we followed the split strategy used in CoOp. 

![image](https://github.com/Mysteriousplayer/KGPT/blob/main/dataset_v1.png)

## Installation
Install all requirements required to run the code on a Python 3.x by:
> First, you need activate a new conda environment.
> 
> pip install -r requirements.txt

## Data processing
All commands should be run under the project root directory. 

```
sh run.sh
```

## Training
After downloading the datasets you need, you can use this command to obtain training samples used in few-shot and easy-to-hard classification task.

```
sh data_processing.sh
```

For example, after obtaining the eth_cars_8.npy (StanfordCars dataset, easy-to-hard classification task, 8/stage), you can copy its path to the corresponding config file (cars_h8.json "new_dir").  
## Results
Results will be saved in log/.  

## Limitations
Our LORE aims to enhance the adaptation of pre-trained ViTs to downstream tasks. Therefore, the primary objective of our experimental design is to validate the effectiveness of pre-trained ViTs in adapting to downstream tasks. To ensure comprehensive evaluations, we compared our LORE with several effective and representative PET methods, including CoOp, Co-CoOp, Maple, ProGrad, Clip-adapter, and VPT. Some of these methods have demonstrated outstanding performance in domain generalization and cross-dataset transfer evaluations using pre-trained CLIP models. However, it is important to note that domain generalization and cross-dataset transfer evaluations assess the CLIP-based model’s generalization ability, which is beyond the scope of this study. We would like to further investigate the generalization ability problem in future work.

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

DINO: https://github.com/facebookresearch/dino

Teaching Matters: Investigating the Role of Supervision in Vision Transformers: https://github.com/mwalmer-umd/vit_analysis

ViG: https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch
