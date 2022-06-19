---
title: "Leveraging Open Source Tools to Deploy Models (Without 😥)"
date: 2022-06-09T20:48:15+08:00
featureImage: images/blog/talk_tfdl_deploying_dl_without_tears/feature_image.gif
postImage: images/blog/talk_tfdl_deploying_dl_without_tears/post_image.png
tags: ["open-source", "deployment"]
categories: ["invited-talks"]
toc: true
socialshare: true
description: "Deploy and share your models to iterate quickly."
images : 
- images/blog/talk_tfdl_deploying_dl_without_tears/post_image.jpeg
---

### 💡 Introduction

{{< figure_resizing src="aboutme.png" >}}


This talk was given to the Tensorflow Deep Learning Malaysia Facebook [group](https://www.facebook.com/groups/TensorFlowMY/) during the June 2022 online meetup.
The group had over 7.5k members consisting of audience from various background related to artificial intelligence in Malaysia.

The goal of the talk is to introduce the members to existing open-source tools they can use to deploy models on the cloud and edge.

Half of the audience has no experience with deep learning. 
Hence, the talk was tailored to beginners in the field.

### 🪂 The Deep Gap
I started the talk by introducing my background as an academic and my experience in the field.

I started exploring the field of deep learning (DL) in 2013.
Having been in the field for over 9+ years now, I shared my stories on how I arrived at this point and my observation of the DL field over the years.

I also shared that being in academia, we are incentivized for publications more than anything else.
As a result, many "groundbreaking" works in DL stopped at the point of publication - which is a pity.
Had the works continue beyond that, they could have the potential to change the industry.

The consequence?

{{% blockquote author="Gartner Survey" %}}
More than 85% of machine learning models fail to make it into production.
{{% /blockquote %}}

I unveiled that the deep gap is that not enough attention is placed on productionizing/deploying DL models in real world applications.

{{< figure_resizing src="meme.jpg" >}}


### ⛏ Technical Walkthrough
I transition the talk to share on some of my recent projects on deploying DL models.

I elaborated on two general categories of deployment environments: 

+ Cloud Deployment. 
+ Edge Deployment.

#### 🌧 Cloud Deployment
Cloud deployment is a setting where the trained DL model is hosted on the cloud infrastructure.

I shared how I trained a state-of-the-art VFNet model with [IceVision](https://github.com/airctic/icevision) and deploy them on an Android phone using the [Hugging Face Hub](https://huggingface.co/docs/hub/index) ecosystem.

The details can be found in the following posts:

+ [Training a Deep Learning Model for Cell Counting in 17 Lines of Code with 17 Images.](https://dicksonneoh.com/portfolio/training_dl_model_for_cell_counting/)

+ [How to Deploy Object Detection Models on Android with Flutter.](https://dicksonneoh.com/portfolio/how_to_deploy_od_models_on_android_with_flutter/)

#### 📱 Edge Deployment
Edge deployment is a setting where the trained DL model is placed on a physical computing hardware (also known as edge device) where the data is collected.

I shared how I trained a state-of-the-art object detection model, [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) to accurately detect license plates on Malaysian vehicles. I also shared how I optimize the model to run 10x faster (at 50 FPS) on a CPU using the [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) toolkit.

I briefly talked about an alternative to the OpenVINO toolkit which can accelerate inference up to 180 FPS using [DeepSparse](https://github.com/neuralmagic/deepsparse) and [SparseML](https://github.com/neuralmagic/sparseml/) library by [Neural Magic](https://neuralmagic.com/).

The details can be found in the following posts:

+ [Faster than GPU: How to 10x your Object Detection Model and Deploy on CPU at 50+ FPS.](https://dicksonneoh.com/portfolio/how_to_10x_your_od_model_and_deploy_50fps_cpu/)

+ [Supercharging YOLOv5: How I Got 182.4 FPS Inference Without a GPU.](https://dicksonneoh.com/portfolio/supercharging_yolov5_180_fps_cpu/)

### 🍧 Takeaways
Here are the takeaways from the brief talk

{{< notice tip >}}

+ Begin with deployment in mind as the end goal.
+ The gap is deeper at the deployment side.
+ Many open-source tools make it easy to deploy models.
+ MLOps - hot topic worth exploring.


{{< /notice >}}


### 📽 Video & Presentation Deck
Recorded video 👇
{{< youtube sVAZevq-8Lc >}}

My presentation [deck](https:&#x2F;&#x2F;www.canva.com&#x2F;design&#x2F;DAFCzWH0RXA&#x2F;view?utm_content=DAFCzWH0RXA&amp;utm_campaign=designshare&amp;utm_medium=embeds&amp;utm_source=link) 👇

<div style="position: relative; width: 100%; height: 0; padding-top: 56.2500%;
 padding-bottom: 48px; box-shadow: 0 2px 8px 0 rgba(63,69,81,0.16); margin-top: 1.6em; margin-bottom: 0.9em; overflow: hidden;
 border-radius: 8px; will-change: transform;">
  <iframe loading="lazy" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; border: none; padding: 0;margin: 0;"
    src="https:&#x2F;&#x2F;www.canva.com&#x2F;design&#x2F;DAFCzWH0RXA&#x2F;view?embed" allowfullscreen="allowfullscreen" allow="fullscreen">
  </iframe>
</div>
