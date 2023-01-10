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
You finally got into a Kaggle competition. You found a *getting-started notebook* written by a Kaggle Grandmaster and immediately trained a state-of-the-art (SOTA) image classification model.

After some fiddling, you found yourself in the leaderboard topping the charts with **99.9851247\% accuracy** on the test set 😎!

Proud of your achievement you reward yourself to some rest and a good night's sleep. 

And then..

{{< figure_resizing src="meme_sleep.jpg" >}}

<!-- I hope this doesn't keep you awake at night like it did for me. -->

With various high level libraries like [Keras](https://keras.io/), [Transformer](https://huggingface.co/docs/transformers/index) and [Fastai](https://www.fast.ai/), the barrier to SOTA models have never been lower.

On top of that with platforms like [Google Colab](https://colab.research.google.com/) and [Kaggle](https://www.kaggle.com/), pretty much anyone can train a reasonably good model using an old laptop or even a mobile phone (with some patience).

**The question is no longer "*can we train a SOTA model?*", but "*what happens after that?*"**

Unfortunately, after getting the model trained, majority data scientists wash their hands off at this point claiming their model works. 
But, what good would SOTA models do if it's just in notebooks and Kaggle leaderboards?

Unless the model is deployed and put to use, it's of little benefit to anyone out there.

{{< figure_resizing src="meme.jpg" >}}

But deployment is painful. Running a model on a mobile phone? 

Forget it 🤷‍♂️.

The frustration is real. I remember spending nights exporting models into `ONNX` and it still fails me.

Mobile deployment doesn't need to be complicated.
In this post I'm going to show you how you can pick from over 600+ SOTA models on [TIMM](https://github.com/rwightman/pytorch-image-models) and deploy them on Android, for free.

<!-- With [TorchScript](https://pytorch.org/docs/stable/jit.html) its possible. -->

{{< notice tip >}}
By the end of this post you will learn how to:
+ Train a SOTA ConvNeXt from TIMM for free on Kaggle.
+ Export the trained model into TorchScript.
+ Create a beaufiful UI and run the model on your Android device with Flutter.

💡**NOTE**: If you already have a [TIMM](https://github.com/rwightman/pytorch-image-models) model feel free to jump straight into [📀 Exporting to TorchScript](https://dicksonneoh.com/portfolio/timm_torchscript_flutter/#-exporting-to-torchscript) section.
{{< /notice >}}


<!-- You might wonder, do I need to learn ONNX? TensorRT? TFLite?

Maybe.

Learning each on of them takes time. Personally, I never had a very positive experience with exporting PyTorch models into ONNX.
It doesn't work every time. -->
<!-- I had to pull my hair over sleepless nights exporting to ONNX.
They are out of the PyTorch ecosystem. -->

<!-- But in this post I will show you solution that holds the best chances of working - TorchScript. -->
<!-- Integrated within the PyTorch ecosystem. -->

But, if you'd like to discover how I train a model using some of the best techniques on Kaggle, read on 👇

### 🥇 PyTorch Image Models

PyTorch Image Models or [TIMM](https://github.com/rwightman/pytorch-image-models) is the open-source computer vision library by [Ross Wightman](https://www.linkedin.com/in/wightmanr/).

The TIMM repository hosts hundreds of recent SOTA models maintained by Ross.
At this point we have 645 pretrained model on TIMM and increasing as we speak.

Other than models TIMM also provides layers, utilities, optimizers, schedulers, data-loaders, augmentations


```bash
pip install timm
```

The TIMM repo also provides training scripts that will let you get SOTA results on your dataset. Feel free to use them to train your model.

In this post I'm going to show you an easy way to train a TIMM model using Fastai 👇


### 🏋️‍♀️ Training with Fastai
[Fastai](https://www.fast.ai/2020/02/13/fastai-A-Layered-API-for-Deep-Learning/) is a deep learning library which provides practitioners with high high-level components that can quickly provide SOTA results.
Under the hood Fastai uses PyTorch but it abstracts away the details and incorporates various best practices in training a model.

```bash
pip install fastai
```

You can access all TIMM models within fastai
It is also possible to search for model architectures using Wildcard as below.

```python
import timm
timm.list_models('*conv*t*')
```

```python
from fastai.vision.all import *

def train(arch, item, batch, epochs=5):
    dls = ImageDataLoaders.from_folder(trn_path, seed=42, valid_pct=0.2, 
                                       item_tfms=item, batch_tfms=batch)
    learn = vision_learner(dls, arch, metrics=error_rate).to_fp16()
    learn.fine_tune(epochs, 0.01)
    return learn

trn_path = path/'train_images'
arch = 'convnext_small_in22k'
learn = train(arch, epochs=12,
              item=Resize((480, 360), method=ResizeMethod.Pad, pad_mode=PadMode.Zeros),
              batch=aug_transforms(size=(256,192), min_scale=0.75))
```

{{< notice tip >}}
View and fork my training notebook [here](https://www.kaggle.com/code/dnth90/timm-at-the-edge).
{{< /notice >}}

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

