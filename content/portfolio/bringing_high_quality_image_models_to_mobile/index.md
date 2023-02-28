---
title: "Bringing High-Quality Image Models to Mobile: HuggingFace TIMM x Android & iOS"
date: 2023-02-27T11:00:15+08:00
featureImage: images/portfolio/bringing_high_quality_image_models_to_mobile/thumbnail.gif
postImage: images/portfolio/bringing_high_quality_image_models_to_mobile/post_image.gif
tags: ["TIMM", "HuggingFace", "paddy-disease", "Flutter", "Android", "iOS", "EdgeNeXt"]
categories: ["deployment", "image-classification", "cloud-ai"]
toc: true
socialshare: true
description: "Discover how HuggingFace's TIMM library enables integration of state-of-the-art computer vision models on mobile."
images : 
- images/portfolio/bringing_high_quality_image_models_to_mobile/post_image.gif
---


### 🌟 Motivation

Meet Bob, a data scientist with a passion for computer vision. Bob had been working on a project to build a model that could identify different types of fruits, from apples to pineapples. He spent countless hours training and fine-tuning the model until it could recognize fruits with 98% accuracy.

Bob even bagged few hackathon awards and praised my many to be the "expert" in computer vision.

Bob was me 5 years ago.

Before joining the industry, I was an academic and researcher in a university in Malaysia.

My favorite part of the job? You've guessed it. Hitting `model.train` on SOTA models and publish them on "reputable journals" just to wash my hands clean and repeat.

{{< figure_resizing src="meme.jpg" >}}

I realized that it doesn't matter how many papers publish about SOTA models, it would not change anything it stays on paper. 

{{% blockquote author="ChatGPT"%}}
Like a painting that remains unseen in an artist's studio, a machine learning model that remains undeployed is a missed opportunity to enrich and enhance the lives of those it was intended to serve.
{{% /blockquote %}}

Why would anyone want to deploy models on mobile devices? 

Here are a few reasons -
+ **Accessibility** - Most people carry their mobile phones with them. A model accessible on mobile devices lets users use models on the go.
+ **Built-in hardware** - Mobile devices comes packaged with on board camera and various sensors. Not worry about integrating new hardware.
+ **User experience** - Enables new form of interaction between apps and sensors on the phone. E.g. computer vision models can be used in an image editing app on the phone.

✅ Yes, for free.

{{< notice tip >}}
⚡ By the end of this post you will learn how to:
+ Upload a SOTA classification model to HuggingFace Spaces and get an inference endpoint.
+ Create a Flutter mobile app that runs on **Android** and **iOS** to call the inference endpoint.
+ Display the inference results on the screen with a beautiful UI.

💡 **NOTE**: Code and data for this post are available on my GitHub repo [here](https://github.com/dnth/huggingface-timm-mobile-blogpost).
{{< /notice >}}

Demo on iOS iPhone 14 Pro

![demo on ios](demo_ios.gif)

Demo on Android - Google Pixel 3 XL.

![demo on android](demo_android.gif)

I've also uploaded the app to Google Playstore. Download and try it out [here](https://play.google.com/store/apps/details?id=com.rice.net).


<!-- Hurdles in mobile computer vision -

+ Limited hardware resources: Mobile devices have limited hardware resources compared to desktop computers or cloud servers. This means that the computational power available on mobile devices may not be enough to run complex computer vision models.

+ Limited memory: Mobile devices also have limited memory, which can make it difficult to store and retrieve large amounts of data required for computer vision models.

+ Battery life: Running complex computer vision models on mobile devices can consume a lot of battery power, which can significantly reduce the battery life of the device.

+ Processing speed: Mobile devices typically have slower processing speeds than desktop computers or cloud servers. This can make it difficult to process large amounts of data required for computer vision models in real-time.

+ Optimization: In order to run computer vision models efficiently on mobile devices, they need to be optimized for the specific hardware and software environment of the device. This requires specialized knowledge and expertise in both computer vision and mobile development.

+ Deployment: Finally, deploying computer vision models on mobile devices requires careful consideration of factors such as app size, download times, and compatibility with different operating systems and devices. -->

Making computer vision models (especially large ones) available on mobile devices sounds interesting in theory.

But in practice there are many hurdles -

+ **Hardware limitation** - Mobile devices usually run on portable hardware with limited processing power, RAM, and battery life. Models needs to be optimized and efficient catering to these limitations.
+ **Optimization** - To put computer vision models on mobile hardware, they usually need to be optimized to run on specific hardware and software environment on the device. This requires specialized knowledge in computer vision and mobile development. 
+ **Practicality** - User experience is a big factor in whether your app will be used by anyone. Nobody wants to use a bloated, slow and inefficient mobile app.

I know that sounds complicated. 
Don't worry because we are **NOT** going to deal with any of that in this blog post!

Enter 👇


### 🤗 HuggingFace x TIMM

<!-- - Introducing HuggingFace and TIMM as a solution
- Introduction to the TIMM (Timm Image Models) library and its architecture
- Advantages of using HuggingFace TIMM for mobile computer vision applications
- Introduction to the HuggingFace Model Hub and its collection of pretrained models
- Explanation of how to use pretrained models with HuggingFace TIMM for mobile computer vision applications
- Comparison of the performance of pretrained models with custom models on mobile devices
- Advantages and disadvantages of using pretrained models -->


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


### ⛄ FAQs
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