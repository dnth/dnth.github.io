---
title: "TIMM at the Edge: Deploying Over 964 PyTorch Image Models on Android with TorchScript and Flutter"
date: 2022-04-18T11:00:15+08:00
featureImage: images/portfolio/timm_torchscript_flutter/thumbnail.gif
postImage: images/portfolio/timm_torchscript_flutter/post_image.png
tags: ["TIMM", "TorchScript", "paddy-disease", "Fastai", "Flutter", "Android"]
categories: ["deployment", "object-classification", "edge"]
toc: true
socialshare: true
description: "Unlocking Over 900 SOTA TIMM models on Android with Torchscript!"
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
⚡ By the end of this post you will learn how to:
+ Train a SOTA model using TIMM and Fastai.
+ Export the trained model into TorchScript.
+ Create a beautiful Flutter app and run the model inference on your Android device.

💡**NOTE**: If you already have a trained [TIMM](https://github.com/rwightman/pytorch-image-models) model, feel free to jump straight into [Exporting to TorchScript](https://dicksonneoh.com/portfolio/timm_torchscript_flutter/#-exporting-to-torchscript) section.
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
At this point we have 964 pretrained model on TIMM and increasing as we speak.

You can install TIMM by simply:
```bash
pip install timm
```

The TIMM repo provides various utility functions and training script. Feel free to use them.
In this post I'm going to show you an easy way to train a TIMM model using Fastai 👇


### 🏋️‍♀️ Training with Fastai
[Fastai](https://www.fast.ai/2020/02/13/fastai-A-Layered-API-for-Deep-Learning/) is a deep learning library which provides practitioners with high high-level components that can quickly provide SOTA results.
Under the hood Fastai uses PyTorch but it abstracts away the details and incorporates various best practices in training a model.

Install Fastai with:
```bash
pip install fastai
```

You can access all TIMM models within Fastai.
For example, we can search for model architectures a [wildcard](https://www.delftstack.com/howto/python/python-wildcard/).
Since we will be running the model on a mobile device, let's search for models that has the word `edge`.

```python
import timm
timm.list_models('*edge*')
```

This outputs all models that match the wildcard.
```bash
['cs3edgenet_x',
 'cs3se_edgenet_x',
 'edgenext_base',
 'edgenext_small',
 'edgenext_small_rw',
 'edgenext_x_small',
 'edgenext_xx_small']
```

Since, we'd run our model on a mobile device, let's select the smallest model available `edgenext_xx_small`.
Now let's use Fastai and quickly train the model.

Firstly import all the necessary packages with
```python
from fastai.vision.all import *
```

Next, load the images into a `DataLoader`.

```python
trn_path = Path('../data/train_images')
dls = ImageDataLoaders.from_folder(trn_path, seed=316, 
                                   valid_pct=0.2, bs=128,
                                   item_tfms=[Resize((224, 224))], 
                                   batch_tfms=aug_transforms(min_scale=0.75))
```

{{< notice note >}}

Parameters for the `from_folder` method:

* `trn_path` -- A `Path` to the training images.
* `valid_pct` -- The percentage of dataset to allocate as the validation set.
* `bs` -- Batch size to use during training.
* `item_tfms` -- Transformation applied to each item.
* `batch_tfms` -- Random transformations applied to each batch to augment the dataset.


{{< /notice >}}

You can show a batch of the images loaded into the `DataLoader` with:

```python
dls.train.show_batch(max_n=8, nrows=2)
```

{{< figure_resizing src="show_batch.png" >}}

Next create a `Learner` object which combines the model and data into one object for training.

```python
learn = vision_learner(dls, 'edgenext_xx_small', metrics=accuracy).to_fp16()
```

Find the best learning rate.

```python
learn.lr_find()
```

{{< figure_resizing src="lr_find.png" >}}

Now train the model.

```python
learn.fine_tune(5, base_lr=1e-2, cbs=[ShowGraphCallback()])
```

{{< figure_resizing src="train.png" >}}


Optionally export the Learner.

```python
learn.export("../../train/export.pkl")
```

{{< notice tip >}}
View and fork my training notebook [here](https://www.kaggle.com/code/dnth90/timm-at-the-edge).
{{< /notice >}}

### 📀 Exporting to TorchScript
Now that we are done training the model, it's time we export the model in a form suiteble on a mobile device.
We can do that easily with [TorchScript](https://pytorch.org/docs/stable/jit.html).

{{% blockquote author="TorchScript Docs" %}}
TorchScript is a way to create serializable and optimizable models from PyTorch code.
{{% /blockquote %}}

All the models on TIMM can be exported to TorchScript with the following code snippet.

```python
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

learn.model.cpu()
learn.model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(learn.model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("model.pt")

```



### 📲 Inference in Flutter

We will be using the [pytorch_lite](https://github.com/zezo357/pytorch_lite) Flutter package.

Supports object classification and detection with TorchScript.

Link to my GitHub [repo](https://github.com/dnth/timm-flutter-pytorch-lite-blogpost).


The screen capture shows the Flutter app in action. The clip runs in real-time and not sped up.

{{< video src="vids/inference_edgenext.mp4" width="400px" loop="true" autoplay="true" muted="true">}}

### 🙏 Comments & Feedback
I hope you've learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter/LinkedIn post or [drop me a message](https://dicksonneoh.com/contact/).
<!-- {{< tweet dicksonneoh7 1534395572022480896>}}


<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:6940225157286264834" height="2406" width="550" frameborder="0" allowfullscreen="" title="Embedded post"></iframe> -->

