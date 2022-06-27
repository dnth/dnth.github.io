---
title: "TIMM at the Edge: How to Deploy 645 PyTorch Image Models on Android with TorchScript and Flutter"
date: 2022-01-09T11:00:15+08:00
featureImage: images/portfolio/timm_torchscript_flutter/thumbnail.gif
postImage: images/portfolio/timm_torchscript_flutter/post_image.png
tags: ["TIMM", "TorchScript", "ConvNeXT", "optimization", "paddy", "Fastai", "Flutter", "Android"]
categories: ["deployment", "object-classification"]
toc: true
socialshare: true
description: "Unlocking 600+ SOTA TIMM models on Android with Torchscript!"
images : 
- images/portfolio/timm_torchscript_flutter/post_image.png
---

{{< notice info >}}
This blog post is still a work in progress. If you require further clarifications before the contents are finalized, please get in touch with me [here](https://dicksonneoh.com/contact/), on [LinkedIn](https://www.linkedin.com/in/dickson-neoh/), or [Twitter](https://twitter.com/dicksonneoh7).
{{< /notice >}}


### 🔥 Motivation
So you know how to train a computer vision model. 
With various packages anyone can train a model.

The question is what happens next?
Do you let the model rot in your notebook? Or do you do something with it?

Majority data scientist wash their hands off at this point claiming my model works, I got SOTA or I won a Kaggle competition.

A SOTA model means little if it's not put to good use in deployment.

{{< figure_resizing src="meme.jpg" >}}

But deploying is painful. Do I need to learn ONNX? TensorRT? TFLite?
Maybe.

Learning each on of them is not easy. I had to pull my hair over sleepless nights exporting to ONNX.
They are out of the PyTorch ecosystem.

Here's a solution that hold the best chances of working - TorchScript.
Integrated within the PyTorch ecosystem.



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


If you already have a TIMM model feel free to jump straight into [Exporting to TorchScript](https://dicksonneoh.com/portfolio/timm_torchscript_flutter/#-exporting-to-torchscript) section.

If you'd want to learn how I train a model using some of the best practices, read on 👇

### 🥇 PyTorch Image Models

PyTorch Image Models or [TIMM](https://github.com/rwightman/pytorch-image-models) is an open source computer vision library by Ross Wightman.

The TIMM repository hosts hundreds of recent SOTA models maintained by Ross.
At this point we have 645 pretrained model on TIMM. Increasing daily.

Other than models TIMM also provides layers, utilities, optimizers, schedulers, data-loaders, augmentations


```bash
pip install timm
```

We can use training script in TIMM and it give you the SOTA results.
TIMM has a lot of advanced techniques, layers, optmizers and so on.

But if you're just exploring and learning, it can take a long time to learn all that. 

If you don't have a lot of time to learn and just want to start training immediately to see how things work, meet Fastai.


### 🏋️‍♀️ Training with Fastai 
Top down learning approach. Learn by doing. You don't need to be a PhD or a "math person".

[Intro to fastai](https://www.fast.ai/2020/02/13/fastai-A-Layered-API-for-Deep-Learning/)

A kid learns to throw a ball by throwing, not learn physics first.

How do we effectively train a model from TIMM?

Fastai has all the best practices that allows you to train any TIMM models and achieve top ranks in Kaggle leaderboards.

Using Jeremy's Kaggle notebook.

Convnext for paddy disease classification.

TIMM model

Top results in leaderboard.

Here is the link to Jeremy's notebook.

Here is the link to my notebook modified from Jeremy's.


### 📀 Exporting to TorchScript

{{% blockquote author="TorchScript Docs" %}}
TorchScript is a way to create serializable and optimizable models from PyTorch code.
{{% /blockquote %}}



All the models on TIMM can be exported to TorchScript


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



### 📲 Inference in Flutter

We will be using the [pytorch_lite](https://github.com/zezo357/pytorch_lite) Flutter package.

Supports object classification and detection with TorchScript.


Link to my GitHub [repo](https://github.com/dnth/timm-flutter-pytorch-lite-blogpost).



{{< video src="vids/inference.mp4" width="400px" loop="true" autoplay="true" muted="true">}}

### 🙏 Comments & Feedback
I hope you've learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter/LinkedIn post or [drop me a message](https://dicksonneoh.com/contact/).
<!-- {{< tweet dicksonneoh7 1534395572022480896>}}


<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:6940225157286264834" height="2406" width="550" frameborder="0" allowfullscreen="" title="Embedded post"></iframe> -->

