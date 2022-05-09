---
title: "How to Compare YOLOX Models with Weights and Biases and Get Your Life Back"
date: 2022-01-07T15:00:15+08:00
featureImage: images/portfolio/comparing_yolox_models_with_weights_and_biases/thumbnail.gif
postImage: images/portfolio/comparing_yolox_models_with_weights_and_biases/post_image.png
tags: ["OpenVINO", "YOLOX", "Wandb", "real-time", "optimization", "license-plate"]
categories: ["deployment", "object-detection", "monitoring"]
toc: true
socialshare: true
description: "Monitor your models with Wandb and get your life back!"
images : 
- images/portfolio/how_to_10x_your_od_model_and_deploy_50fps_cpu/post_image.png
---
{{< notice info >}}
This blog post is still a work in progress. If you require further clarifications before the contents are finalized, please get in touch with me [here](https://dicksonneoh.com/contact/), on [LinkedIn](https://www.linkedin.com/in/dickson-neoh/), or [Twitter](https://twitter.com/dicksonneoh7).
{{< /notice >}}

### 🔎 Motivation

{{< notice tip >}}
By the end of this post you will learn how to:
+ Install the Wandb client and log the YOLOX training metrics to Wandb.
+ Compare training metrics on Wandb dashboard.
+ Picking the best model from mAP and FPS values.
{{< /notice >}}

"**So many models, so little time!**"

As a machine learning engineer, I often hear this phrase thrown around in many variations.

In object detection alone, there are already several hundreds of models out there. 
With each passing day, better models are added as new discoveries are made.
If you're new, this can easily get overwhelming.

Even within the YOLOX series there are 6 variations of the model to choose from.
There are 3 questions that beg answering:

+ How do you pick the best model?
+ How do the models compare to one another?
+ How do you keep track and log the performance of each model?

In this blog post I will show you how I accomplish all of them by using a free and simple tool from [Weights and Biases](https://wandb.ai/home) (Wandb) 👇

**PS**: No Excel sheets involved.

### 📉 Wandb - Google Drive for Machine Learning


Life is short they say. So why waste it on monitoring your deep learning models when you can automate them?
This is what Wandb is trying to solve. It's like Google Drive for machine learning.

Wandb helps individuals and teams build models faster.
With just few lines of code, you can compare models, log important metrics, and collaborate with teammates.
It's free to get started. Click [here](https://wandb.ai/) to create an account. 

{{< figure_resizing src="wandb.png">}}

This post is a sequel to my previous post where I showed [how to deploy YOLOX models on CPU at 50 FPS](https://dicksonneoh.com/portfolio/how_to_10x_your_od_model_and_deploy_50fps_cpu/).
This time around I will show you how I get the most from the YOLOX models by logging the performance metrics and comparing them on Wandb.

Let's first install the Wandb client for `Python`:

``` bash
pip install wandb
```

Next, run 
```bash
wandb login
```
from your terminal to authenticate your machine. The API key is stored in `~/.netrc`.

### 👀 Monitoring Training Metrics
Using with YOLOX.
Monitoring metrics.
mAP.

### ⚖️ Comparing YOLOX Models 
Convert into INT8.
Comparison TLDR.
FPS.
Visual inspection

### ⛳️ Wrapping up
Simple easy free.

### 🙏 Comments & Feedback
I hope you've learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter post or [drop me a message](https://dicksonneoh.com/contact/).
{{< tweet dicksonneoh7 1521342853640728576>}}


If you like what you see and don't want to miss any of my future contents, follow me on Twitter and LinkedIn where I deliver more of these tips in bite-size posts.
