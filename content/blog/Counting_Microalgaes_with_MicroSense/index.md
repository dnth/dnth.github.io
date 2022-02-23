---
title: "MicroSense - A deep learning powered cell counting app with Flutter"
date: 2022-02-14T15:07:15+08:00
featureImage: images/blog/Counting_Microalgaes_with_MicroSense/feature_image.gif
postImage: images/blog/Counting_Microalgaes_with_MicroSense/post_image.png
tags: ["Flutter", "HuggingFace", "Android"]
categories: ["deployment", "training"]
toc: true
socialshare: true
---

### Motivation
Numerous biology and medical procedure involve counting cells from images taken with microscope.
Counting cells reveals the concentration of bacteria and viruses and gives vital information on the progress of a disease.
Uses hemocytometer or a counting chamber.
Creates volumetric grid for accurate counting. Done properly can be accurate. Don't require sophisticated instrument can be easily performed.
Image shows a sample image from a hemocytometer.

{{< figure_resizing src="hemocytometer.jpg" >}}

Microbiologist often count cells manually this is a method known as direct counting.

This process can be tedious and prone to mistakes.
Why not automate the counting using a trained deep learning model?
This blog post showcases MicroSense, a proof-of-concept product that can automate the counting of microalgae cells or any cells for that matter using deep learning and computer vision.

In this blog post I will teach you how to:
1. Train a deep learning model with IceVision to count microalgae cells.
2. Deploy the model on a mobile app with Flutter and Hugging Face Spaces.

{{< figure_resizing src="microsense_logo.png" link="https://play.google.com/store/apps/details?id=com.micro.sense">}}

### Model Development with IceVision
Figure illustrates the raw detection of cells from microscope image. The model is a RetinaNet with a ResNet50 backbone trained using [IceVision](https://github.com/airctic/icevision).
{{< figure_resizing src="detection.png" >}}

#### Preparing the dataset
Collected from a local pond in Malaysia.

Augmented using imgaug

Sample training dataset
{{< figure_resizing src="dataset_sample.png" >}}


#### Training
IceVision computer vision framework.

#### Local Inference
Inference on a local machine
{{< figure_resizing src="inference.png" >}}

### Hosting model on Hugging Face Spaces
Deploying a large deep learning model on mobile may not be the most effective way.
App will be big, inference will be slow due to the lightweight mobile processor.

Alternative to use a remote server to infer and send the results via API calls.
Hugging Face provides free and easy way of doing that. 
Each Spaces environment is limited to 16GB RAM and 8 CPU cores.

Exposes HTTP API to run inference.

This section shows how you can host the model on Hugging Face Spaces and use the API for inferencing.
Checkout the Space [here](https://huggingface.co/spaces/dnth/webdemo-microalgae-counting).
Checkout the exposed API [here](https://hf.space/gradioiframe/dnth/webdemo-microalgae-counting/api).

### Deploying on Android

Figure shows an Android app written using the Flutter framework. The inference is done by calling the API from our Space.
Checkout the app published on Google Playstore [here](https://play.google.com/store/apps/details?id=com.micro.sense).
{{< figure src="microsense.gif" >}}

#### Setting up Flutter

#### Remote Inference

#### Conclusion
Github [repo](https://github.com/dnth/webdemo-microalgae-detection)