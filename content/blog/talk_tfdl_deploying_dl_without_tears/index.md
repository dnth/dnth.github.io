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

I unveiled that the deep gap is that not enough attention is placed on deploying deep learning models in real world applications.

{{< figure_resizing src="meme.jpg" >}}


### ⛏ Technical Walkthrough
I transition the talk to share on some of my recent projects on deploying DL models.
Two general categories of deployment environments are cloud and edge deployment.

#### 🌧 Cloud Deployment
Cloud deployment is a setting where the trained DL model is hosted on the cloud infrastructure.

I shared how I trained a state-of-the-art VFNet model with IceVision and deploy them on an Android phone using the Hugging Face Hub ecosystem.

The details can be found in the following posts:

+ [Training a Deep Learning Model for Cell Counting in 17 Lines of Code with 17 Images.](https://dicksonneoh.com/portfolio/training_dl_model_for_cell_counting/)

+ [How to Deploy Object Detection Models on Android with Flutter.](https://dicksonneoh.com/portfolio/how_to_deploy_od_models_on_android_with_flutter/)

#### 📱 Edge Deployment
Edge deployment is a setting where the trained DL model is place on a physical hardware where the data is collected.

### 🍧 Takeaways

{{< notice tip >}}

+ Begin with deployment in mind as the end goal.
+ The gap is deeper at the deployment side.
+ MLOps - hot topic.
+ Many open-source tools make it easy to deploy models.

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
