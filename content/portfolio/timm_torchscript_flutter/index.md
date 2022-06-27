---
title: "TIMM at the Edge: How to Deploy 645 PyTorch Image Models on Android with TorchScript and Flutter"
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
SOTA models usually take a lot of resources to train and deploy.

Putting complicated model on mobile is painful.

What if you can do it completely for free?

I'm going to show you how you can put a SOTA model on an Android phone easily.

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

Other than models TIMM also provides layers, utilities, optimizers, schedulers, data-loaders, augmentations


```bash
pip install timm
```


### 🏋️‍♀️ Training with Fastai and TIMM

How do we effectively train a model from TIMM?

Fastai has all the best practices that allows you to train any TIMM models and achieve top ranks in Kaggle leaderboards.

Using Jeremy's Kaggle notebook.

Convnext for paddy disease classification.

TIMM model

Top results in leaderboard.


### 📀 Exporting to TorchScript

{{% blockquote author="TorchScript Docs" %}}
TorchScript is a way to create serializable and optimizable models from PyTorch code.
{{% /blockquote %}}



All the models on TIMM can be exported to TorchScript

### 📲 Inference in Flutter

We will be using the [pytorch_lite](https://github.com/zezo357/pytorch_lite) Flutter package.

Supports object classification and detection with TorchScript.

Preparing the model

```python
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torch.load('model_scripted.pt',map_location="cpu")
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("model.pt")

```

{{< video src="vids/inference.mp4" width="400px" loop="true" autoplay="true" muted="true">}}

### 🙏 Comments & Feedback
I hope you've learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter/LinkedIn post or [drop me a message](https://dicksonneoh.com/contact/).
<!-- {{< tweet dicksonneoh7 1534395572022480896>}}


<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:6940225157286264834" height="2406" width="550" frameborder="0" allowfullscreen="" title="Embedded post"></iframe> -->

