---
title: "Counting microalgaes with MicroSense"
date: 2022-02-14T15:07:15+08:00
featureImage: images/blog/counting-microalgaes-with-microsensee/feature_image.gif
postImage: images/blog/counting-microalgaes-with-microsensee/detection.png
tags: ["Flutter"]
categories: ["deep learning"]
---

### Introduction
Microbiologist count cells manually.
This process can be tedious and prone to mistakes.
Why not automate the counting using a trained deep learning model?
This blog post showcases MicroSense, a proof of concept product that can automate the counting of microalgae cells or any cells for that matter using deep learning and computer vision.

{{< figure src="/images/portfolio/microsense/microsense_logo.png" caption="A proof of concept product." width=750 link="https://play.google.com/store/apps/details?id=com.micro.sense">}}

### Training
Figure illustrates the raw detection of cells from microscope image. The model is a RetinaNet with a ResNet50 backbone trained using [IceVision](https://github.com/airctic/icevision).
{{< figure src="/images/portfolio/microsense/detection.png" width=750 >}}

### Hosting
This section shows how you can host the model on Hugging Face Spaces and use the API for inferencing.
Checkout the Space [here](https://huggingface.co/spaces/dnth/webdemo-microalgae-counting).
Checkout the exposed API [here](https://hf.space/gradioiframe/dnth/webdemo-microalgae-counting/api).

### Deploying on Android
Figure shows an Android app written using the Flutter framework. The inference is done by calling the API from our Space.
Checkout the app published on Google Playstore [here](https://play.google.com/store/apps/details?id=com.micro.sense).
{{< figure src="/images/portfolio/microsense/microsense.gif" height=750 >}}
