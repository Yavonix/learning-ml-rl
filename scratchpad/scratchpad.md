## Self-Supervised Approaches

Invariance-based pretraining methods (representation should stay the same even when the input changes in some irrelevant)
- SimCLR
- JEPA

Generative-based pretraining methods (learn by trying to generate or reconstruct part/all of the input)
- MAE Masked autoencoder

Problem with invariance based:
- SimCLR needs hand-designed augmentations: crop, colour jitter, blur, etc.
- Can introduce bias if those transformations impact the data itself
- **Augmentations encode human assumptions**
- Teaches the model to ignore superficial details and keep the stable semantic content
- Not so good for example in medical imaging where subtle colour/intensity/texture differences matter

Problem with generative based:
- Just needs masking/corruption, so it is more generic
- often learn lower-level representations (texture, colour, local detail rather than concepts like cat, dog, car)
- Often underperform invariance-based methods on simple representation tests like linear probing

JEPA is trying to get the benefits of invariance-based SSL without manually specifying the invariances through hand-crafted augmentations. (without using extra prior knowledge encoded through image transformations)

## High Level Overview

Representation collapse refers to a model learning a trivial representation that satisfies the training objective but no useful information
- Complete collapse occurs when a model maps all inputs to the same representation vector
- Dimensional collapse occurs when a model maps only uses a few dimensions

Reconstruction loss is a training objective that compares a model's reconstructed input with the original input, forcing the model to preserve enough information to recreate the data

In JEPA, removing reconstruction loss creates a risk of representation collapse

Current methods to avoid representation collapse:
0. Just include reconstruction loss (forces the representation to preserve enough information to reconstruct the input)
1. EMA: Use weights in target encoder that are an exponential moving average (EMA) of the weights in other encoder (I-JEPA, V-JEPA, DINO, BYOL)
2. Infomax: (information maximisation; try to make the learned representation contain as much useful information as possible)
   1. Sample-contrastive methods: (SimCLR, Siamese nets, DrLIM, etc) tend not to work well in high dimension, to require large batches, and hard negative mining
   2. Dimension-contrastive methods: (Barlow Twins, VICReg, SIGReg/ LeJEPA, MMCR, MCR2, etc) make embedding dimensions decorrelated


A. SSL by reconstruction/prediction doesn't work for high-dim, continuous,  noisy data
B. EMA sucks: no loss function being minimized,  requirement for weightmsharing....
C. Sample-contrastive informax doesn't scale to high dimension
D. My money is on dimension-contrastive methods like SIGReg/LeJEPA


[LeCun](https://x.com/ylecun/status/2007907701989232684)
```
I think you missed the main ideas.
- The basic premise of JEPA is that training by reconstructio/prediction in input space is evil (or counterproductive). The details are almost always unpredictable.  Hence prediction must take place in representation space, where unpredictable details are eliminated.
- The main issue with JEPA is how to prevent collapse (in the absence of reconstruction loss). There are two classes of methods: 
(1) EMA: Using weights in target encoder that are an exponential moving average (EMA) of the weights in other encoder (I-JEPA, V-JEPA, DINO, BYOL).
(2) Infomax: Using a regularizer that attempts to maximize the information content of the representation (e.g. over a batch). There are two sets of methods for that:
(2a) sample-contrastive methods: that want to make each representation vector different from the others (Siamese nets, DrLIM, SimCLR, etc). They tend to not work well in high dimension, to require large batches, and hard negative mining
(2b) dimension-contrastive methods: that want to make each variable independent from the others (Barlow Twins, VICReg, SIGReg/ LeJEPA, MMCR, MCR2....)
Bottom line: 
A. SSL by reconstruction/prediction doesn't work for high-dim, continuous,  noisy data
B. EMA sucks: no loss function being minimized,  requirement for weightmsharing....
C. Sample-contrastive informax doesn't scale to high dimension
D. My money is on dimension-contrastive methods like SIGReg/LeJEPA
```












