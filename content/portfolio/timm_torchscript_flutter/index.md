---
title: "PyTorch at the Edge: How I Deploy SOTA Image Model on Android with TorchScript and Flutter"
date: 2022-01-09T11:00:15+08:00
featureImage: images/portfolio/timm_torchscript_flutter/thumbnail.gif
postImage: images/portfolio/timm_torchscript_flutter/post_image.png
tags: ["TIMM", "TorchScript", "ConvNeXT", "optimization", "paddy", "Fastai", "Flutter", "Android"]
categories: ["deployment", "object-classification"]
toc: true
socialshare: true
description: "Deploy any TIMM models on Android with Torchscript!"
images : 
- images/portfolio/timm_torchscript_flutter/post_image.png
---

{{< notice info >}}
This blog post is still a work in progress. If you require further clarifications before the contents are finalized, please get in touch with me [here](https://dicksonneoh.com/contact/), on [LinkedIn](https://www.linkedin.com/in/dickson-neoh/), or [Twitter](https://twitter.com/dicksonneoh7).
{{< /notice >}}


### 🔥 Motivation
SOTA models usually take a lot of resources to run.

Putting complicated model on mobile is painful.

With [TorchScript](https://pytorch.org/docs/stable/jit.html) its possible.

{{< notice tip >}}
By the end of this post you will learn how to:
+ Train a SOTA ConvNext from TIMM for free on Kaggle.
+ Export the trained model into TorchScript.
+ Create a beaufiful UI and run the model on your Android device with Flutter.
{{< /notice >}}

### 🥇 PyTorch Image Models

PyTorch Image Models or [TIMM](https://github.com/rwightman/pytorch-image-models) is an open source computer vision library by Ross Wightman.

The TIMM repository hosts hundreds of recent SOTA models maintained by Ross.



### 🏋️‍♀️ Training with Fastai and TIMM
Using Jeremy's Kaggle notebook.

Convnext for paddy disease classification.

TIMM model

Top results in leaderboard.


### 📀 Exporting to TorchScript

All the models on TIMM can be exported to TorchScript

### 📲 Inference in Flutter

{{< video src="vids/inference.mp4" width="400px" loop="true" autoplay="true" muted="true">}}

### 🙏 Comments & Feedback
I hope you've learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter/LinkedIn post or [drop me a message](https://dicksonneoh.com/contact/).
<!-- {{< tweet dicksonneoh7 1534395572022480896>}}


<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:6940225157286264834" height="2406" width="550" frameborder="0" allowfullscreen="" title="Embedded post"></iframe> -->

