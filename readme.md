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
  <img src="assets/inference.gif" alt="Inference demo" width="300">
</p>

<a id="about"></a>

# 💡 About

Openworld is an open-source world model based heavily on Google's original [Genie paper](https://arxiv.org/abs/2402.15391). It was trained on data gathered from the VizDoom library, and can run action-conditioned gameplay inference on a single high-end GPU There are two versions of the model within this repository; Openworld-10M and Openworld-28M. the 10 million parmeter model was trained on the vizdoom-takecover environment with only 3 actions, and the 28 million parameter model was trained on vizdoom-deathmatch, a much more complex environment with 7 actions.

## Motivations

When I first came across the demo for [Genie 3](https://deepmind.google/models/genie/), I was immediately awestruck. This was the first time I truly realized that generative AI had the potential to accurately model the dynamics of the real world, rather than being just a tool for making funny Will Smith videos. My goal with this project was, first of all, to learn as much as I could about a new frontier of AI - world models. On top of that, I also wanted to "democratize" as much of my learnings as possible for other people interested in the same things, so I've created quite an extensive write-up on the architecture and techniques I used.

<a id="architecture"></a>

# 🏛️ Architecture

<a id="training"></a>

# 🏋️ Training

<a id="inference"></a>

# 🎮 Inference

<a id="experimentation"></a>

# 🧪 Experimentation