# DAROAT
This repository provides the architecture of the diffusion assisted reconstruction method (DAR) developed to increase the quality of optoacoustic tomography images.

Our proposal is to use a conditional diffusion model to improve the image quality reconstructed with a standard and well-proven method. The main idea is to use the image reconstructed by this well-proven method as conditional information to a diffusion model that will enhance the final image and eliminate possible artifacts besides increasing resolution (if desired).  Three major blocks can be identified in the proposed method: 

1) the initial well-proved reconstruction method,
2) the conditional information preprocessing (CIP), and
3) the conditional diffusion model in reduced dimension.

In Fig. 1 a depiction of the different blocks and their interconnections is presented

![plot](./images/architecturev4.png)

##### Figure 1. The forward diffusion process only acts during training phase. 3-tuples (V,K,Q) indicate the multi-head cross-attention mechanisms at each UNet's scales. Sinusoidal positional time embeddings are used for representing the time-steps in the reverse diffusion process.

### 1. Initial reconstruction method

The measured sinogram is processed by matrix $\mathbf{A}^T$ generating an initial image corresponding to the LBP [Arridge_2016](https://arxiv.org/abs/1602.02027). This method is simple to implement and numerically efficient but usually introduces some artifacts. For this reason, this initial image is fed into an appropriate neural network, with a sufficiently large expressive capacity power, in order to correct artifacts and improve quality. This neural network is trained  separately to the other blocks in our system. In particular, we chose a Fully-Dense UNet (FD-UNet) architecture, which is well-known to be a competitive solution for OAT image reconstruction [Gonzalez_2023](https://arxiv.org/abs/2210.08099). 

```
FDUNet(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
  (db1): DenseBlock(
    (dl1): DenseLayer(
      (conv1): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((32, 128, 128), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((8, 128, 128), eps=1e-05, elementwise_affine=True)
    )
    (dl2): DenseLayer(
      (conv1): Conv2d(40, 32, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((32, 128, 128), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((8, 128, 128), eps=1e-05, elementwise_affine=True)
    )
    (dl3): DenseLayer(
      (conv1): Conv2d(48, 32, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((32, 128, 128), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((8, 128, 128), eps=1e-05, elementwise_affine=True)
    )
    (dl4): DenseLayer(
      (conv1): Conv2d(56, 32, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((32, 128, 128), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((8, 128, 128), eps=1e-05, elementwise_affine=True)
    )
  )
  (db2): DenseBlock(
    (dl1): DenseLayer(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((64, 64, 64), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((16, 64, 64), eps=1e-05, elementwise_affine=True)
    )
    (dl2): DenseLayer(
      (conv1): Conv2d(80, 64, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((64, 64, 64), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((16, 64, 64), eps=1e-05, elementwise_affine=True)
    )
    (dl3): DenseLayer(
      (conv1): Conv2d(96, 64, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((64, 64, 64), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((16, 64, 64), eps=1e-05, elementwise_affine=True)
    )
    (dl4): DenseLayer(
      (conv1): Conv2d(112, 64, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((64, 64, 64), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((16, 64, 64), eps=1e-05, elementwise_affine=True)
    )
  )
  (db3): DenseBlock(
    (dl1): DenseLayer(
      (conv1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((128, 32, 32), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((32, 32, 32), eps=1e-05, elementwise_affine=True)
    )
    (dl2): DenseLayer(
      (conv1): Conv2d(160, 128, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((128, 32, 32), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((32, 32, 32), eps=1e-05, elementwise_affine=True)
    )
    (dl3): DenseLayer(
      (conv1): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((128, 32, 32), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((32, 32, 32), eps=1e-05, elementwise_affine=True)
    )
    (dl4): DenseLayer(
      (conv1): Conv2d(224, 128, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((128, 32, 32), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((32, 32, 32), eps=1e-05, elementwise_affine=True)
    )
  )
  (db4): DenseBlock(
    (dl1): DenseLayer(
      (conv1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((256, 16, 16), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((64, 16, 16), eps=1e-05, elementwise_affine=True)
    )
    (dl2): DenseLayer(
      (conv1): Conv2d(320, 256, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((256, 16, 16), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((64, 16, 16), eps=1e-05, elementwise_affine=True)
    )
    (dl3): DenseLayer(
      (conv1): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((256, 16, 16), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((64, 16, 16), eps=1e-05, elementwise_affine=True)
    )
    (dl4): DenseLayer(
      (conv1): Conv2d(448, 256, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((256, 16, 16), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((64, 16, 16), eps=1e-05, elementwise_affine=True)
    )
  )
  (db5): DenseBlock(
    (dl1): DenseLayer(
      (conv1): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((512, 8, 8), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((128, 8, 8), eps=1e-05, elementwise_affine=True)
    )
    (dl2): DenseLayer(
      (conv1): Conv2d(640, 512, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((512, 8, 8), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((128, 8, 8), eps=1e-05, elementwise_affine=True)
    )
    (dl3): DenseLayer(
      (conv1): Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((512, 8, 8), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((128, 8, 8), eps=1e-05, elementwise_affine=True)
    )
    (dl4): DenseLayer(
      (conv1): Conv2d(896, 512, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((512, 8, 8), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((128, 8, 8), eps=1e-05, elementwise_affine=True)
    )
  )
  (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (up1): UpBlock(
    (up): ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=(2, 2))
    (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), padding=same)
  )
  (db6): DenseBlock(
    (dl1): DenseLayer(
      (conv1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((256, 16, 16), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((64, 16, 16), eps=1e-05, elementwise_affine=True)
    )
    (dl2): DenseLayer(
      (conv1): Conv2d(320, 256, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((256, 16, 16), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((64, 16, 16), eps=1e-05, elementwise_affine=True)
    )
    (dl3): DenseLayer(
      (conv1): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((256, 16, 16), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((64, 16, 16), eps=1e-05, elementwise_affine=True)
    )
    (dl4): DenseLayer(
      (conv1): Conv2d(448, 256, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((256, 16, 16), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((64, 16, 16), eps=1e-05, elementwise_affine=True)
    )
  )
  (up2): UpBlock(
    (up): ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2))
    (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), padding=same)
  )
  (db7): DenseBlock(
    (dl1): DenseLayer(
      (conv1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((128, 32, 32), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((32, 32, 32), eps=1e-05, elementwise_affine=True)
    )
    (dl2): DenseLayer(
      (conv1): Conv2d(160, 128, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((128, 32, 32), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((32, 32, 32), eps=1e-05, elementwise_affine=True)
    )
    (dl3): DenseLayer(
      (conv1): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((128, 32, 32), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((32, 32, 32), eps=1e-05, elementwise_affine=True)
    )
    (dl4): DenseLayer(
      (conv1): Conv2d(224, 128, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((128, 32, 32), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((32, 32, 32), eps=1e-05, elementwise_affine=True)
    )
  )
  (up3): UpBlock(
    (up): ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2))
    (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), padding=same)
  )
  (db8): DenseBlock(
    (dl1): DenseLayer(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((64, 64, 64), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((16, 64, 64), eps=1e-05, elementwise_affine=True)
    )
    (dl2): DenseLayer(
      (conv1): Conv2d(80, 64, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((64, 64, 64), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((16, 64, 64), eps=1e-05, elementwise_affine=True)
    )
    (dl3): DenseLayer(
      (conv1): Conv2d(96, 64, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((64, 64, 64), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((16, 64, 64), eps=1e-05, elementwise_affine=True)
    )
    (dl4): DenseLayer(
      (conv1): Conv2d(112, 64, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((64, 64, 64), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((16, 64, 64), eps=1e-05, elementwise_affine=True)
    )
  )
  (up4): UpBlock(
    (up): ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
    (conv1): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1), padding=same)
  )
  (db9): DenseBlock(
    (dl1): DenseLayer(
      (conv1): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((32, 128, 128), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((8, 128, 128), eps=1e-05, elementwise_affine=True)
    )
    (dl2): DenseLayer(
      (conv1): Conv2d(40, 32, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((32, 128, 128), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((8, 128, 128), eps=1e-05, elementwise_affine=True)
    )
    (dl3): DenseLayer(
      (conv1): Conv2d(48, 32, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((32, 128, 128), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((8, 128, 128), eps=1e-05, elementwise_affine=True)
    )
    (dl4): DenseLayer(
      (conv1): Conv2d(56, 32, kernel_size=(1, 1), stride=(1, 1), padding=same)
      (conv2): Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=same)
      (relu): ReLU()
      (lnorm1): LayerNorm((32, 128, 128), eps=1e-05, elementwise_affine=True)
      (lnorm2): LayerNorm((8, 128, 128), eps=1e-05, elementwise_affine=True)
    )
  )
  (conv2): Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=same)
)
```

### 2. Conditional information preprocessing (CIP)

The initial reconstructed image from the previous block will be used as conditional information for the diffusion process. This information is passed to the diffusion model through a trainable function $g_\theta$. In our case, we considered $g_\theta$ as the encoder from a vanilla autoencoder.

```
VanAE(
  (encoder): Sequential(
    (0): Sequential(
      (0): Linear(in_features=4096, out_features=3072, bias=True)
      (1): ReLU()
      (2): Dropout(p=0.0, inplace=False)
    )
    (1): Sequential(
      (0): Linear(in_features=3072, out_features=2048, bias=True)
      (1): ReLU()
      (2): Dropout(p=0.0, inplace=False)
    )
    (2): Linear(in_features=2048, out_features=1024, bias=True)
  )
  (decoder): Sequential(
    (0): Sequential(
      (0): Linear(in_features=1024, out_features=2048, bias=True)
      (1): ReLU()
      (2): Dropout(p=0.0, inplace=False)
    )
    (1): Sequential(
      (0): Linear(in_features=2048, out_features=3072, bias=True)
      (1): ReLU()
      (2): Dropout(p=0.0, inplace=False)
    )
    (2): Linear(in_features=3072, out_features=4096, bias=True)
  )
)
```

### 3. Conditional diffusion model in reduced dimension

The conditional information corresponding to the initial image reconstruction patches is introduced in the reverse diffusion process. The cost function is then optimized using backpropagation techniques. Each step $t=1,\dots, T$ in the reverse process is modeled by a UNet architecture. We considered the default parameters implemented in the library [Diffusers](https://huggingface.co/docs/diffusers/api/models/unet2d-cond). We only modified the number of output channel at each scale with respect to the default ones to the following values (the same for each time-step $t$): (128, 256, 512, 1024). This was done for optimizing the use of GPU memory during training. Also, as we are not working with RGB images, the number of channels at the input and output for the UNet were set to 1.

```
UNet2DConditionModel(
  (conv_in): Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (time_proj): Timesteps()
  (time_embedding): TimestepEmbedding(
    (linear_1): Linear(in_features=128, out_features=512, bias=True)
    (act): SiLU()
    (linear_2): Linear(in_features=512, out_features=512, bias=True)
  )
  (encoder_hid_proj): Linear(in_features=1024, out_features=768, bias=True)
  (down_blocks): ModuleList(
    (0): CrossAttnDownBlock2D(
      (attentions): ModuleList(
        (0-1): 2 x Transformer2DModel(
          (norm): GroupNorm(32, 128, eps=1e-06, affine=True)
          (proj_in): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): Linear(in_features=128, out_features=128, bias=False)
                (to_k): Linear(in_features=128, out_features=128, bias=False)
                (to_v): Linear(in_features=128, out_features=128, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=128, out_features=128, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): Linear(in_features=128, out_features=128, bias=False)
                (to_k): Linear(in_features=768, out_features=128, bias=False)
                (to_v): Linear(in_features=768, out_features=128, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=128, out_features=128, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=128, out_features=1024, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=512, out_features=128, bias=True)
                )
              )
            )
          )
          (proj_out): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 128, eps=1e-05, affine=True)
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=128, bias=True)
          (norm2): GroupNorm(32, 128, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (1): CrossAttnDownBlock2D(
      (attentions): ModuleList(
        (0-1): 2 x Transformer2DModel(
          (norm): GroupNorm(32, 256, eps=1e-06, affine=True)
          (proj_in): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): Linear(in_features=256, out_features=256, bias=False)
                (to_k): Linear(in_features=256, out_features=256, bias=False)
                (to_v): Linear(in_features=256, out_features=256, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=256, out_features=256, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): Linear(in_features=256, out_features=256, bias=False)
                (to_k): Linear(in_features=768, out_features=256, bias=False)
                (to_v): Linear(in_features=768, out_features=256, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=256, out_features=256, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=256, out_features=2048, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=1024, out_features=256, bias=True)
                )
              )
            )
          )
          (proj_out): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 128, eps=1e-05, affine=True)
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=256, bias=True)
          (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 256, eps=1e-05, affine=True)
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=256, bias=True)
          (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (2): CrossAttnDownBlock2D(
      (attentions): ModuleList(
        (0-1): 2 x Transformer2DModel(
          (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
          (proj_in): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): Linear(in_features=512, out_features=512, bias=False)
                (to_k): Linear(in_features=512, out_features=512, bias=False)
                (to_v): Linear(in_features=512, out_features=512, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=512, out_features=512, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): Linear(in_features=512, out_features=512, bias=False)
                (to_k): Linear(in_features=768, out_features=512, bias=False)
                (to_v): Linear(in_features=768, out_features=512, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=512, out_features=512, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=512, out_features=4096, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=2048, out_features=512, bias=True)
                )
              )
            )
          )
          (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 256, eps=1e-05, affine=True)
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=512, bias=True)
          (norm2): GroupNorm(32, 512, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 512, eps=1e-05, affine=True)
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=512, bias=True)
          (norm2): GroupNorm(32, 512, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (3): DownBlock2D(
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 512, eps=1e-05, affine=True)
          (conv1): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=1024, bias=True)
          (norm2): GroupNorm(32, 1024, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 1024, eps=1e-05, affine=True)
          (conv1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=1024, bias=True)
          (norm2): GroupNorm(32, 1024, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
    )
  )
  (up_blocks): ModuleList(
    (0): UpBlock2D(
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 2048, eps=1e-05, affine=True)
          (conv1): Conv2d(2048, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=1024, bias=True)
          (norm2): GroupNorm(32, 1024, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): ResnetBlock2D(
          (norm1): GroupNorm(32, 1536, eps=1e-05, affine=True)
          (conv1): Conv2d(1536, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=1024, bias=True)
          (norm2): GroupNorm(32, 1024, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(1536, 1024, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (1): CrossAttnUpBlock2D(
      (attentions): ModuleList(
        (0-2): 3 x Transformer2DModel(
          (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
          (proj_in): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): Linear(in_features=512, out_features=512, bias=False)
                (to_k): Linear(in_features=512, out_features=512, bias=False)
                (to_v): Linear(in_features=512, out_features=512, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=512, out_features=512, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): Linear(in_features=512, out_features=512, bias=False)
                (to_k): Linear(in_features=768, out_features=512, bias=False)
                (to_v): Linear(in_features=768, out_features=512, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=512, out_features=512, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=512, out_features=4096, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=2048, out_features=512, bias=True)
                )
              )
            )
          )
          (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 1536, eps=1e-05, affine=True)
          (conv1): Conv2d(1536, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=512, bias=True)
          (norm2): GroupNorm(32, 512, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(1536, 512, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 1024, eps=1e-05, affine=True)
          (conv1): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=512, bias=True)
          (norm2): GroupNorm(32, 512, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): ResnetBlock2D(
          (norm1): GroupNorm(32, 768, eps=1e-05, affine=True)
          (conv1): Conv2d(768, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=512, bias=True)
          (norm2): GroupNorm(32, 512, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (2): CrossAttnUpBlock2D(
      (attentions): ModuleList(
        (0-2): 3 x Transformer2DModel(
          (norm): GroupNorm(32, 256, eps=1e-06, affine=True)
          (proj_in): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): Linear(in_features=256, out_features=256, bias=False)
                (to_k): Linear(in_features=256, out_features=256, bias=False)
                (to_v): Linear(in_features=256, out_features=256, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=256, out_features=256, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): Linear(in_features=256, out_features=256, bias=False)
                (to_k): Linear(in_features=768, out_features=256, bias=False)
                (to_v): Linear(in_features=768, out_features=256, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=256, out_features=256, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=256, out_features=2048, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=1024, out_features=256, bias=True)
                )
              )
            )
          )
          (proj_out): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 768, eps=1e-05, affine=True)
          (conv1): Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=256, bias=True)
          (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 512, eps=1e-05, affine=True)
          (conv1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=256, bias=True)
          (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): ResnetBlock2D(
          (norm1): GroupNorm(32, 384, eps=1e-05, affine=True)
          (conv1): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=256, bias=True)
          (norm2): GroupNorm(32, 256, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (3): CrossAttnUpBlock2D(
      (attentions): ModuleList(
        (0-2): 3 x Transformer2DModel(
          (norm): GroupNorm(32, 128, eps=1e-06, affine=True)
          (proj_in): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): Linear(in_features=128, out_features=128, bias=False)
                (to_k): Linear(in_features=128, out_features=128, bias=False)
                (to_v): Linear(in_features=128, out_features=128, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=128, out_features=128, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): Linear(in_features=128, out_features=128, bias=False)
                (to_k): Linear(in_features=768, out_features=128, bias=False)
                (to_v): Linear(in_features=768, out_features=128, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=128, out_features=128, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=128, out_features=1024, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=512, out_features=128, bias=True)
                )
              )
            )
          )
          (proj_out): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 384, eps=1e-05, affine=True)
          (conv1): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=128, bias=True)
          (norm2): GroupNorm(32, 128, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        (1-2): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 256, eps=1e-05, affine=True)
          (conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=512, out_features=128, bias=True)
          (norm2): GroupNorm(32, 128, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
  )
  (mid_block): UNetMidBlock2DCrossAttn(
    (attentions): ModuleList(
      (0): Transformer2DModel(
        (norm): GroupNorm(32, 1024, eps=1e-06, affine=True)
        (proj_in): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=1024, out_features=1024, bias=False)
              (to_k): Linear(in_features=1024, out_features=1024, bias=False)
              (to_v): Linear(in_features=1024, out_features=1024, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1024, out_features=1024, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=1024, out_features=1024, bias=False)
              (to_k): Linear(in_features=768, out_features=1024, bias=False)
              (to_v): Linear(in_features=768, out_features=1024, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1024, out_features=1024, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=1024, out_features=8192, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=4096, out_features=1024, bias=True)
              )
            )
          )
        )
        (proj_out): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (resnets): ModuleList(
      (0-1): 2 x ResnetBlock2D(
        (norm1): GroupNorm(32, 1024, eps=1e-05, affine=True)
        (conv1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=512, out_features=1024, bias=True)
        (norm2): GroupNorm(32, 1024, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
      )
    )
  )
  (conv_norm_out): GroupNorm(32, 128, eps=1e-05, affine=True)
  (conv_act): SiLU()
  (conv_out): Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
```

##### Soon we will be adding the implementation code of the architecture detailed above.
