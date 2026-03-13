# ![](assets/banner.png)

<p align="center">
  <a href="#about">About</a> &nbsp;&bull;&nbsp;
  <a href="#architecture">Architecture</a> &nbsp;&bull;&nbsp;
  <a href="#training">Training</a> &nbsp;&bull;&nbsp;
  <a href="#inference">Inference</a> &nbsp;&bull;&nbsp;
  <a href="#experimentation">Experimentation</a>
</p>

<br>

<p align="center">
  <img src="assets/inference.gif" alt="Inference demo" width="200">
</p>

<a id="about"></a>

# 💡 About

Openworld is an open-source world model based heavily on Google's original [Genie paper](https://arxiv.org/abs/2402.15391). It was trained on data gathered from the VizDoom library, and can run action-conditioned gameplay inference on a single high-end GPU. 

There are two versions of the model within this repository; Openworld-10M and Openworld-28M. the 10 million parameter model was trained on the vizdoom-takecover environment with only 3 actions, and the 28 million parameter model was trained on vizdoom-deathmatch, a much more complex environment with 7 actions.

## Motivations

When I first came across the demo for [Genie 3](https://deepmind.google/models/genie/), I was immediately awestruck. This was the first time I truly realized that generative AI had the potential to accurately model the dynamics of the real world, rather than being just a tool for making funny Will Smith videos. 

My goal with this project was, first of all, to learn as much as I could about a new frontier of AI - world models. On top of that, I also wanted to share my learnings  with other people interested in the same things, so I've created a write-up on the architecture and techniques I used.

<a id="architecture"></a>

# 🏛️ Architecture

## Introduction

Genie (and thereby Openworld) is broken into 3 seperate models; the video tokenizer, the latent action model, and the dynamics model. 

<p align="center"><img src="assets/architecture-overview.png" width="600"></p>

The Video Tokenizer is responsible for converting raw video into what are called latent tokens; compressed, low-dimensional representations of data. The Dynamics Model takes these latent tokens, alongside action tokens, and generates a set of latent tokens representing the next frame. 

How does it get these action tokens? In training, the Latent Action Model takes raw video and learns to generate action tokens, representing some meaningful change between previous latents and the next latents. The Dynamics Model then conditions its generation on these tokens, teaching the model what kinds of actions lead to what kinds of outcomes. Before we dive into the architecture of these models, there are some concepts I'd like to introduce first:

### Finite Scalar Quantization

Finite Scalar Quantization (FSQ) is a technique for quantizing vectors. I'd like to thank [Anand Majmudar](https://x.com/Almondgodd/status/1971314294517350533) and his very cool and similar project for introducing me to this concept. This is such a cool and simple algorithm which made my life a lot easier.

Essentially, Genie works with quantized tokens/vectors. They utilized an algorithm called the Vector-Quantized Variational Auto-Encoder. This approach maintains a codebook of vectors. To quantize a given input vector, we simply return the closest codebook vector. These codebook vectors are also learned; they are updated via some algorithm (gradient descent, exponential moving average). 

The problem with this algorithm is that it suffers from something called [codebook collapse](https://machinelearning.wtf/terms/codebook-collapse/), where only a small fraction of codebook vectors are actually utilized (due to the nature of pushing vectors around during training). FSQ solves that by re-thinking the entire quantization space. Instead of learned vectors, FSQ codebook vectors are essentially evenly-spaced points along the edges of a hypercube. We quantize in a similar fashion; essentially normalizing and snapping an input vector to one of these points. With a latent vector dimension of 3, this quantization space resembles a cube:

<p align="center"><img src="assets/fsq.png" width="300"></p>

<a id="training"></a>

# 🏋️ Training

TBD

<a id="inference"></a>

# 🎮 Inference

TBD

<a id="experimentation"></a>

# 🧪 Experimentation

TBD