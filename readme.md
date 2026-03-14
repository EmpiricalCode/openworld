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

Openworld is an open-source transformer world model based heavily on Google's original [Genie paper](https://arxiv.org/abs/2402.15391). It was trained on data gathered from the VizDoom library, and can run action-conditioned gameplay inference on a single high-end GPU. 

There are two versions of the model within this repository; Openworld-10M and Openworld-28M. the 10 million parameter model was trained on the vizdoom-takecover environment with only 3 actions, and the 28 million parameter model was trained on vizdoom-deathmatch, a much more complex environment with 7 actions.

## Motivations

When I first came across the demo for [Genie 3](https://deepmind.google/models/genie/), that was when I truly realized that generative AI had the potential to accurately model the dynamics of the real world, rather than being just a tool for making funny Will Smith videos. My goal with this project was, first of all, to learn as much as I could about a new frontier of AI - world models. On top of that, I also wanted to share my learnings  with other people interested in the same things, so I've created a write-up on the architecture and techniques I used.

<a id="architecture"></a>

# 🏛️ Architecture

## Introduction

Genie (and thereby Openworld) is broken into 3 seperate models; the video tokenizer, the latent action model, and the dynamics model. 

<p align="center"><img src="assets/architecture-overview.png" width="600"></p>

The Video Tokenizer is responsible for converting raw video into what are called latent tokens; compressed, low-dimensional representations of data. The Dynamics Model takes these latent tokens, alongside action tokens, and generates a set of latent tokens representing the next frame. 

How does it get these action tokens? In training, the Latent Action Model takes raw video and learns to generate action tokens, representing some meaningful change between previous latents and the next latents. The Dynamics Model then conditions its generation on these tokens, teaching the model what kinds of actions with what kinds of previous latents lead to what kinds of future latents. Before we dive into the architecture of these models, there are some concepts I'd like to introduce first:

### Spatio-Temporal Transformer

This transformer architecture is the bread and butter of this world model (note that for this section, a basic understanding of transformers is a pre-requisite). It consists of a chain of spatio-temporal transformer blocks. Each block contains a spatial attention block, a temporal attention block, and then a feed-forward layer. As described in the Genie paper, the spatial attention block attends over all patch tokens within a frame. The temporal attention block then attends over all patch tokens in "tubelets" through time (IE, tokens within the same spatial location, through all frames in time).

<p align="center"><img src="assets/spatio-temporal-transformer.png" width="600"></p>

A lot of the finer details are ommitted from this diagram that I will briefly go over. The FFN layer uses a SwiGLU activation. At first I tried GeLU, and later switched to SwiGLU since it seems to perform better in LLMs. First, note that each attention block implements standard Multi-Headed Attention (MHA). We also employ residual connections on the attention blocks and the FFN block, which is SoTA for transformers. For layer norm, we have two options. For usecases which don't require conditioning, we use RMSNorm. For the latent action model and the dynamics model, which need to condition generation on action tokens, we use AdaLN.

I recommend checking out [this](https://arxiv.org/abs/2102.05095) paper, which describes that divided self attention performs better in video classification tasks. This was likely the basis for the spatio-temporal transformer.

### Finite Scalar Quantization

Finite Scalar Quantization (FSQ) is a technique for quantizing vectors. I'd like to thank [Anand Majmudar](https://x.com/Almondgodd/status/1971314294517350533) and his very cool and similar project for introducing me to this concept. This is such a cool and simple algorithm which made my life a lot easier.

Essentially, Genie works with quantized latents (vectors). They utilized an algorithm called the Vector-Quantized Variational Auto-Encoder (VQ-VAE). This approach maintains a codebook of vectors. To quantize a given input vector, we simply return the closest codebook vector. These codebook vectors are also learned; they are updated via some algorithm (gradient descent, exponential moving average). 

The problem with this algorithm is that it suffers from something called [codebook collapse](https://machinelearning.wtf/terms/codebook-collapse/), where only a small fraction of codebook vectors are actually utilized (due to the nature of pushing vectors around during training). For example, some vectors may be pushed into regions that the inputs do not visit.

<p align="center"><img src="assets/fsq.png" width="300"></p>

FSQ solves that by re-thinking the entire quantization space. Instead of learned vectors, FSQ codebook vectors are essentially evenly-spaced points along the edges of a hypercube. We quantize in a similar fashion; essentially snapping an input vector to one of these points (scaled within -1 to 1). The reason this works is that, since the vectors are evenly spaced and don't move around, they all have an equal chance of being utilized. With a latent vector dimension of 3 and 2 bins, this quantization space resembles the corners of a cube, as shown above.

Check out the FSQ paper [here](https://arxiv.org/abs/2309.15505).

### Embedding

How do we actually feed our images into the tokenizer and action model? We need some way of embedding images into sequences of tokens that our transformers can actually work with. The answer is patch embeddings. We split a video into a sequence of individual frames. For each frame, we divide it into patches. For each patch, we flatten all the R, G, and B values into a single vector (each patch goes from shape (P, P, 3) into (P * P * 3)). 

<p align="center"><img src="assets/embedding.png" width="800"></p>

Next, we project all of these patch vectors into the embedding space (the dimensionality of the actual embedding tokens our transformer will work with). This is done through a single linear layer, which is effectively a matrix multiplication across all patch vectors in order to convert them into tokens (dimensionality P * P * 3 -> embed_dim).

Finally, we need some way to encode *position* into each token. Just like in a sentence, a token in one position has a different contextual meaning than if it were somewhere else. To achieve this, we add 3 positional encoding vectors onto each patch token that encodes where that patch is in terms of time, X, and Y position. I won't go too into detail for positional encoding, since there is a great resource I found [here](https://huggingface.co/blog/designing-positional-encoding).

## Video Tokenizer

The video tokenizer is fundamentally an auto-encoder which includes an FSQ step in between the encoder and decoder. It works by compressing (encoding) video into a latent representation, then trying to decode that into the same video again, learning to preserve the most useful information within its latent.

<p align="center"><img src="assets/tokenizer-encoder.png" width="800"></p>

The encoder takes raw video, runs it through patch embedding and positional encoding blocks, and a spatio-temporal transformer. Finally, the tokens are projected down into the latent dimension, from the embedding dimension. This is where the data compression occurs.

<p align="center"><img src="assets/tokenizer-quantization.png" width="300"></p>

Next, we pass the pre-quantized latent through FSQ. Note that the resulting quantized latent is the video tokenizer's output, which is used by the dynamics model.

<p align="center"><img src="assets/tokenizer-decoder.png" width="800"></p>

Finally, our decoder takes the quantized latents, passes it through the spatio-temporal transformer again, and then unembeds each patch back into pixel formats. In hindsight, I also could have added a positional encoding block before we pass the tokens into the spatio-temporal transformer within the decoder, but in practice the video tokenizer never struggled with its objective.


## Latent Action Model

WIP Section

## Dynamics Model

WIP Section
