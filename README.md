# DAROAT
This repository provides the architecture of the diffusion assisted reconstruction method (DAR) developed to increase the quality of optoacoustic tomography images.

The proposed method architecture is presented in the following figure:

![plot](./images/architecturev4.png)

The forward diffusion process only acts during training phase. 3-tuples (V,K,Q) indicate the multi-head cross-attention mechanisms at each UNet's scales. Sinusoidal positional time embeddings are used for representing the time-steps in the reverse diffusion process.
