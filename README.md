# TGANet: Text-guided attention for improved polyp segmentation

## 1. Abstract
<div align="justify">
Colonoscopy is a gold standard procedure but is highly operator-dependent. Automated polyp segmentation, a precancerous precursor, can minimize missed rates and timely treatment of colon cancer at an early stage. Even though there are deep learning methods developed for this task, variability in polyp size can impact model training, thereby limiting it to the size attribute of the majority of samples in the training dataset that may provide sub-optimal results to differently sized polyps. In this work, we exploit <i>size-related</i> and <i>polyp number-related</i> features in the form of text attention during training. We introduce an auxiliary classification task to weight the text-based embedding that allows network to learn additional feature representations that can distinctly adapt to differently sized polyps and can adapt to cases with multiple polyps. Our experimental results demonstrate that these added text embeddings improve the overall performance of the model compared to state-of-the-art segmentation methods. We explore four different datasets and provide insights for size-specific improvements. Our proposed <i>text-guided attention network</i> (TGANet) can generalize well to variable-sized polyps in different datasets.
  </div>

## 2. Architecture
<img src="images/TGANet-architecture.png">

## 3. Implementation
The proposed architecture is implemented using the PyTorch framework (1.9.0+cu111) with a single GeForce RTX 3090 GPU of 24 GB memory. 

### 3.1 Dataset
We have used the following datasets:
- [Kvasir-SEG](https://datasets.simula.no/downloads/kvasir-seg.zip)
- [CVC-ClinicDB](https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar?dl=0)
- [BKAI](https://www.kaggle.com/competitions/bkai-igh-neopolyp/data)
- [Kvasir-Sessile](https://datasets.simula.no/downloads/kvasir-sessile.zip)

All the dataset follows an 80:10:10 split for training, validation and testing, except for the Kvasir-SEG, where the dataset is split into training and testing. 

### 3.2 Weight file
You can download the weight file from the following links:
- [Kvasir-SEG](https://drive.google.com/file/d/1kVbwYNj2h_K15uBI4DpH8GHbqJM4Csdd/view?usp=sharing)
- [CVC-ClinicDB](https://drive.google.com/file/d/1fJnfX91zTFMlC--O5bH-7kYcueS5eBGo/view?usp=sharing)
- [BKAI](https://drive.google.com/file/d/1_Z2Uj4zZwcsx_If1qY34PoImZzjRV-XG/view?usp=sharing)
- [Kvasir-Sessile](https://drive.google.com/file/d/1vP8y5vMF_LlNgoXSAOmv2uXwujeIx6zC/view?usp=sharing)


## 4. Quantative Results
<img src="images/Quantative.png">

## 5. Qualitative Results
<img src="images/TGA-PolySeg-Qualitative.jpg">

## 6. Citation
<pre>
@inproceedings{tomar2022tganet
title={TGANet: Text-guided attention for improved polyp segmentation},
author={Tomar, Nikhil Kumar and Jha, Debesh and Bagci, Ulas and Ali, Sharib},
booktitle={arXiv preprint arXiv:2205.04280},
year={2022}
} 
</pre>

## 7. License
The source code is free for research and education use only. Any comercial use should receive a formal permission from the first author.

## 8. Contact
Please contact nikhilroxtomar@gmail.com for any further questions. 
