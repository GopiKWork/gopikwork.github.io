---
layout: post
title: "microdiffusion: Image Diffusion in 300 Lines of Pure Python"
date: 2026-03-24
description: "A single-file, zero-dependency implementation of the DDPM algorithm — the same core behind Stable Diffusion, DALL-E, and Midjourney"
tags: [diffusion-models, machine-learning, python, generative-ai]
---

## TL;DR

One Python file. 300 lines. Zero dependencies. Trains a diffusion model on 8×8 images and generates new ones using DDPM with classifier-free guidance. Inspired by Karpathy's [microgpt](https://karpathy.github.io/2026/02/12/microgpt/).

## Intro

Andrej Karpathy's [microgpt](https://karpathy.github.io/2026/02/12/microgpt/) distills a GPT into 200 lines of dependency-free Python. It strips language models to their algorithmic core: autograd, attention, next-token prediction. I wanted to do the same for image generation. microdiffusion is the result.

One file, 300 lines, zero dependencies. It trains a diffusion model on 62 hand-crafted 8×8 images and generates new ones using the same algorithm that powers Stable Diffusion. The autograd engine is borrowed directly from microgpt. Everything else from noise schedules, forward diffusion, denoising networks and classifier-free guidance (CFG) is specific to this.

You can view the code in [GitHub](https://github.com/gopikwork/microdiffusion).

## Core Diffusion Problem

Language models answer one question: "given these tokens, what token comes next?" Diffusion models answer a different one: "given this noisy image, what does the clean version look like?"

The training process works backward from intuition. First, take a clean image and destroy it by adding Gaussian noise over some timesteps (e.g. T=20). Then train a neural network to predict the noise that was added. At generation time, start from pure random noise and iteratively subtract the predicted noise, step by step, until a clean image emerges.

This is the entire algorithm. The rest of this post walks through the implementation. After 1500 training steps (~15 minutes on a laptop CPU), the model generates 8×8 shapes rendered as ASCII art and the solution doesn't have any image libraries.

![Microdiffusion model components](/assets/images/microdiffusion/model-components.png)

## Dataset

Production diffusion models train on billions of image-text pairs. microdiffusion uses 62 hand-crafted 8×8 binary images stored in a text file: digits 0–9, letters A-Z, and geometric patterns (checkerboard, stripes, diamond, heart, arrows).

Each image is a grid of `#` (black, 0.0) and `.` (white, 1.0):

```
# 3
.#####..
......#.
......#.
..####..
......#.
......#.
.#####..
........
```

The `load_dataset()` function reads this file and converts each image into a flat list of 64 floats. One image = 64 numbers. That's the entire input representation.

## Autograd from microgpt

Training requires gradients. For each of the 50,688 parameters, we need to know how nudging that parameter affects the loss. The `Value` class from microgpt handles it. This is the same autograd engine from microgpt. There is one modification though.

microgpt uses recursive topological sort for backpropagation. Diffusion models build deeper computation graphs (64 pixels × multiple layers per training step), which exceed Python's default recursion limit of 1000. This is replaced with an iterative stack.

```python
def backward(self):
    topo, visited = [], set()
    stack = [(self, False)]
    while stack:
        v, processed = stack.pop()
        if processed:
            topo.append(v)
            continue
        if v in visited:
            continue
        visited.add(v)
        stack.append((v, True))
        for child in v._children:
            if child not in visited:
                stack.append((child, False))
    self.grad = 1
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad
```

After `backward()` completes, every `Value` in the graph has a `.grad` containing the derivative of the loss with respect to that value. This is what PyTorch's `loss.backward()` does, just on scalars instead of tensors.

Lets review the core components briefly.

## Forward diffusion

Image values are added with noise generated based on the timesteps.

The cosine noise schedule controls how quickly signal is destroyed. The alpha value generated with this scheduler essentially determines how much of the original image is preserved when noise is added. Constant s (value of 0.008) prevents α̅(0) from being exactly 1.

The cosine noise schedule defines how much of the original image signal remains at each timestep:

$\bar{\alpha}(t) = \cos^2\left(\frac{\frac{t}{T} + s}{1 + s} \cdot \frac{\pi}{2}\right)$

Where:
- $t$ is the current timestep (0 to $T$)
- $T = 20$ is the total number of timesteps
- $s = 0.008$ is a small offset to prevent $\bar{\alpha}(0)$ from being exactly 1.0

Given a clean image x₀ and timestep t, gaussian noise ε ~ N(0,1), forward diffusion produces a noisy version in one step.

$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$

Where:
- $x_0$ is the clean image (64 pixel values)
- $\epsilon$ is random Gaussian noise (64 random numbers from a bell curve centered at 0)
- $\bar{\alpha}_t$ controls the mix ratio

This doesn't require iterative noise additions. As we pre-calculated α̅(t), we can directly go from clean image to noise at any step level.

![Forward diffusion process - adding noise across timesteps](/assets/images/microdiffusion/forward-diffusion.png)

```
--- forward diffusion visualization (digit '3' being destroyed by noise) ---
t=0 (clean):
  @ @ @ @ @
                @
               @
       @ @ @ @
                @
               @
     @ @ @ @ @

t=5:
- @ + @ # @   .
       : - . @ :
     .   :   + :
   - @ # + @
       *   . . = #
   : : * : .   @
     * @ -   @   =
 * -   =   : = .

t=10:
@ @ % @ @ @ - +
 @ @   @   = *
   - -   @   @ + -
 @   @ @ - # #
   *     - = :   +
 :   * : + : @ =
   @ @ @ % * + #
 -   = @       *

t=15:
@ @ @ @   @   @
   @ +   @   @ #
 @ : @   - . @ @
 . @ @ = = @ @ @
     @   @ - @
   # @   = @ - @
     @ + @ @
       - @     @ @ @

t=20:
  @ @ @ @ % : %
 @ %     @ .   .
 @ @   @ - - @ %
 -   - *   @ * @
 *   +   # + @ @
 @ @ @         :
       @ - @   .
 @   % @ @ @ *
```

## Denoising Network

Large diffusion models such as Stable Diffusion, use U-Nets with hundreds of millions of parameters. We use a 3-layer MLP with 50,688 parameters. However, the algorithm is similar in concept.

The network receives three pieces of information joined end-to-end: noise at timestep t, timestep embedding and label (e.g. Checkerboard) embedding.

![Denoising Network](/assets/images/microdiffusion/denoising-network.png)

Their values are concatenated to produce a vector. The network sees all three simultaneously i.e what the noisy image looks like (input image), how noisy it is (from timestep value), and what it should be generating (predicted noise).

The time embedding encodes the timestep using sinusoidal functions at different frequencies. 


```python
def time_embedding(t):
    emb = []
    for i in range(TIME_EMB_DIM // 2):  # 8 iterations for dim=16
        freq = 1.0 / (10.0 ** (i / 8))
        emb.append(math.sin(t * freq))
        emb.append(math.cos(t * freq))
    return emb  # returns 16 floats
```

The network needs to know how noisy the input is. We encode the timestep $t$ into a 16-dimensional vector using sine and cosine waves at different frequencies. This is similar to positional encoding in transformer architecture. High frequencies distinguish nearby timesteps (t=5 vs t=6). 

For `t=5`, the 16-dimensional time embedding vector looks like:

```
freq[0] = 1.000:  sin(5.0)=-0.959,  cos(5.0)= 0.284
freq[1] = 0.316:  sin(1.6)= 0.999,  cos(1.6)=-0.029
freq[2] = 0.100:  sin(0.5)= 0.479,  cos(0.5)= 0.878
freq[3] = 0.032:  sin(0.2)= 0.158,  cos(0.2)= 0.987
freq[4] = 0.010:  sin(0.05)=0.050,  cos(0.05)=0.999
freq[5] = 0.003:  sin(0.02)=0.016,  cos(0.02)=1.000
freq[6] = 0.001:  sin(0.005)=0.005, cos(0.005)=1.000
freq[7] = 0.000:  sin(0.002)=0.002, cos(0.002)=1.000
```

Result: `[-0.959, 0.284, 0.999, -0.029, 0.479, 0.878, 0.158, 0.987, 0.050, 0.999, 0.016, 1.000, 0.005, 1.000, 0.002, 1.000]`

For `t=6`, the high-frequency components change significantly (sin(6)=−0.279 vs sin(5)=−0.959) while low-frequency components barely change. This is the same idea as **positional encoding** in Transformers.


Low frequencies distinguish far-apart ones (t=1 vs t=20).

The label embedding is different. It is a learned 16-dimensional vector per label. This is the "micro version" of the CLIP embeddings used in Stable Diffusion. Instead of encoding "a photo of a cat" into 768 dimensions, we encode "heart" into 16 dimensions.

```python
'label_emb': matrix(num_labels, LABEL_EMB_DIM, std=0.1)  # 62 × 16 matrix
```

Each of the 62 labels gets its own 16-dimensional vector, initialized as small random numbers. These are learned during training. Backpropagation adjusts them until each label's embedding captures what makes that pattern unique.

When `label_idx=None` (unconditional), the label embedding is all zeros. network gets no information about what the image should be.


The core network is made of three fully connected layers with a skip connection. The skip connection transforms the 96-dimensional input directly to 128 dimensions and adds it to layer 2's output before ReLU. This is similar to the residual connection in microgpt. Parameters are updated using the Adam optimizer with a learning rate that decays linearly from 100% to 10% of the initial value.

Output dimension is same as the image dimension with 64 values. One predicted noise value per pixel. No activation function, because noise can be any real number. Another small difference is that microgpt has separate training and inference code paths.

## DDPM

Training is based on DDPM (Denoising Diffusion Probabilistic Models). It's the foundational algorithm that made diffusion models practical for image generation. It formalized the training process into two steps: 1/ diffusion step — this adds noise to the cleaner image for a given timestep which determines how much noise needs to be added (controlled by α̅). 2/ Reverse process predicts the noise at a given timestep so that we can remove the noise from the diffusion process and attempt to extract the original image.

As far as the training objective is concerned, it is very simple. It is essentially sample a clean image, add noise, ask the network to predict the noise, and minimize the squared error (MSE).

```python
loss = sum((pred - Value(true)) ** 2 for pred, true in zip(noise_pred, noise_true))
loss = loss * (1.0 / IMG_DIM)
```

Each of the 1500 training steps do the following:

- Select a random image from training dataset and its label
- Select a random timestep t ∈ [1, 20]
- Add noise to the image at that timestep (forward diffusion)
- Ask the network to predict the noise (Denoising network)
- Compute MSE between predicted and actual noise
- Backpropagate and update parameters with Adam

## Sampling process

Start from pure Gaussian noise. For each timestep from T=20 down to 1, predict the noise and subtract it.

The fresh noise z at each step (except t=1) makes sampling stochastic — each run produces a different image. At t=1, we skip the noise for a clean final output.

When generating with a label, we run the network twice per timestep:

```python
noise_cond   = denoise_network(x, t, label_idx)   # "what noise, knowing it's a heart?"
noise_uncond = denoise_network(x, t, None)         # "what noise, with no label?"
noise_pred   = noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond)
```

The difference (noise_cond − noise_uncond) points toward the label that was supplied. Multiplying by GUIDANCE_SCALE (default value 3.0), amplifies that direction. This is called Classifier Free Guidance (CFG). Training randomly drops the label 10% of the time, so the network learned both conditional and unconditional predictions.

![Sampling Process](/assets/images/microdiffusion/sampling-process.png)

## Running

```bash
python microdiffusion.py
```

It is simple. No pip install. No dependencies. ~15 minutes on a modern laptop. The output shows training progress, then unconditional samples, then label-guided samples.

## Quick summary of steps

**Training:**
1. Take a clean image and its label
2. Destroy it with a random amount of noise
3. Ask the network: "what noise did I add?" (sometimes with the label, sometimes without)
4. Compare its answer to the truth (MSE loss)
5. Adjust the network and label embeddings to be less wrong
6. Repeat 1500 times

**Generating (unconditional):**
1. Start with pure random noise
2. Ask the network: "what noise is in this?" (no label)
3. Remove some of that predicted noise
4. Repeat from $t=T$ down to $t=1$
5. Result: a random image that looks like the training data

**Generating (conditional with CFG):**
1. Start with pure random noise
2. Ask the network twice: once with the label, once without
3. Amplify the difference (the "direction toward the label")
4. Remove the guided noise prediction
5. Repeat from $t=T$ down to $t=1$
6. Result: an image that matches the requested label


## Comparison

This is not a replacement for real image generation. It produces ASCII characters, not true images. The output is noisy, but the code demonstrates the complete process. As you can see from below, there is a significant difference in terms of the scale of production grade image generation system.

![Comparison table](/assets/images/microdiffusion/comparison.png)

This is a toy example to demonstrate the core building blocks of a diffusion model.

## References

- [Andrej Karpathy Blog on microgpt](https://karpathy.github.io/2026/02/12/microgpt/)
- [Denoising Diffusion Probabilistic Models Paper](https://arxiv.org/abs/2006.11239)
- [Improved DDPM — Nichol & Dhariwal](https://arxiv.org/abs/2102.09672)

---

*Thank you for taking the time to read. The views expressed in this article are my own and do not necessarily represent the views of my employer. Reach me at [LinkedIn](https://www.linkedin.com/in/gopinathk/).*
