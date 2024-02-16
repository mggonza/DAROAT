# DAROAT
This repository provides the architecture of the diffusion assisted reconstruction method (DAR) developed to increase the quality of optoacoustic tomography images.

Our proposal is to use a conditional diffusion model to improve the image quality reconstructed with a standard and well-proven method. The main idea is to use the image reconstructed by this well-proven method as conditional information to a diffusion model that will enhance the final image and eliminate possible artifacts besides increasing resolution (if desired).  Three major blocks can be identified in the proposed method: 

1) the initial well-proved reconstruction method,
2) the conditional information preprocessing (CIP), and
3) the conditional diffusion model in reduced dimension.

In Fig. 1 a depiction of the different blocks and their interconnections is presented

![plot](./images/architecturev4.png)

##### Figure 1. The forward diffusion process only acts during training phase. 3-tuples (V,K,Q) indicate the multi-head cross-attention mechanisms at each UNet's scales. Sinusoidal positional time embeddings are used for representing the time-steps in the reverse diffusion process.
