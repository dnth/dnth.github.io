---
title: "Bringing High-Quality Image Models to Mobile: HuggingFace TIMM Meets Android & iOS"
date: 2023-02-27T11:00:15+08:00
featureImage: images/portfolio/pytorch_at_the_edge_timm_torchscript_flutter/thumbnail.gif
postImage: images/portfolio/pytorch_at_the_edge_timm_torchscript_flutter/post_image.gif
tags: ["TIMM", "HuggingFace", "paddy-disease", "Flutter", "Android", "iOS", "EdgeNeXt"]
categories: ["deployment", "image-classification", "cloud-ai"]
toc: true
socialshare: true
description: "Discover how HuggingFace's TIMM library enables integration of state-of-the-art computer vision models on mobile."
images : 
- images/portfolio/pytorch_at_the_edge_timm_torchscript_flutter/post_image.gif
---


### 🌟 Motivation

Meet Bob, a data scientist with a passion for computer vision. Bob had been working on a project to build a model that could identify different types of fruits, from apples to pineapples. He spent countless hours training and fine-tuning the model until it could recognize fruits with 98% accuracy.

Bob even bagged few hackathon awards and praised my many to be the "expert" in computer vision.

Bob was me 5 years ago.

Before joining the industry, I was an academic and researcher in a university in Malaysia.

My favorite part of the job? You've guessed it. Hitting `model.train` on SOTA models and publish them on some journal just to wash my hands clean and repeat.


- Brief overview of computer vision and its importance in mobile applications
- The challenge of bringing high-quality image models to mobile devices
- The goal of the article and a preview of what will be covered

Hurdles in mobile computer vision -

+ Limited hardware resources: Mobile devices have limited hardware resources compared to desktop computers or cloud servers. This means that the computational power available on mobile devices may not be enough to run complex computer vision models.

+ Limited memory: Mobile devices also have limited memory, which can make it difficult to store and retrieve large amounts of data required for computer vision models.

+ Battery life: Running complex computer vision models on mobile devices can consume a lot of battery power, which can significantly reduce the battery life of the device.

+ Processing speed: Mobile devices typically have slower processing speeds than desktop computers or cloud servers. This can make it difficult to process large amounts of data required for computer vision models in real-time.

+ Optimization: In order to run computer vision models efficiently on mobile devices, they need to be optimized for the specific hardware and software environment of the device. This requires specialized knowledge and expertise in both computer vision and mobile development.

+ Deployment: Finally, deploying computer vision models on mobile devices requires careful consideration of factors such as app size, download times, and compatibility with different operating systems and devices.


✅ Yes, for free.

{{< notice tip >}}
⚡ By the end of this post you will learn how to:
+ Upload a SOTA classification model to HuggingFace Spaces and get an inference endpoint.
+ Create a functional mobile app that runs on Android and iOS to call the inference endpoint.
+ Display the inference results on the screen with a beautiful UI.

💡 **NOTE**: Code and data for this post are available on my GitHub repo [here](https://github.com/dnth/huggingface-timm-mobile-blogpost).
{{< /notice >}}


Demo on iOS iPhone 14 Pro

![demo on ios](demo_ios.gif)


### 🤗 HuggingFace x TIMM

- Introducing HuggingFace and TIMM as a solution
- Introduction to the TIMM (Timm Image Models) library and its architecture
- Advantages of using HuggingFace TIMM for mobile computer vision applications
- Introduction to the HuggingFace Model Hub and its collection of pretrained models
- Explanation of how to use pretrained models with HuggingFace TIMM for mobile computer vision applications
- Comparison of the performance of pretrained models with custom models on mobile devices
- Advantages and disadvantages of using pretrained models


### 📥 Hosting a Model on HuggingFace Hub

### 🔄 Inference Endpoint
- Rest API with Gradio.

### 📲 Flutter
- Build user interface.
- Don't want to get user lost in the detail implementation. Refer to GitHub repo.

### 🎄 Conclusion and Future of Mobile Computer Vision

- Summary of the importance of high-quality image models for mobile applications
- Recap of HuggingFace TIMM's role in bringing these models to mobile devices
- Discussion of future possibilities for mobile computer vision using HuggingFace TIMM and other emerging technologies


## ⛄ FAQs
- What is computer vision, and why is it important for mobile applications?
- What is HuggingFace, and how does it relate to computer vision?
- What is the TIMM library, and what makes it unique compared to other computer vision libraries?
- What are the limitations of Android and iOS for computer vision applications?
- What is the Android Neural Networks API (NNAPI), and how does it work with HuggingFace TIMM?
- What is the Core ML framework, and how does it work with HuggingFace TIMM?
- What are pretrained models, and why are they important in computer vision?
- How do I use pretrained models with HuggingFace TIMM for mobile computer vision applications?
- How does the performance of pretrained models compare to custom models on mobile devices?
- What are the advantages and disadvantages of using pretrained models for mobile computer vision applications?