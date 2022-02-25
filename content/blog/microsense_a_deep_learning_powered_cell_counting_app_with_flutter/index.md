---
title: "MicroSense - A deep learning powered cell counting app with Flutter"
date: 2022-02-24T15:07:15+08:00
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
To accomplish the counting, a device known as the hemocytometer or a counting chamber is used.
The hemocytometer creates volumetric grid to divide each region on the image for accurate counting. 
The following image shows an image taken from a hemocytometer.

{{< figure_resizing src="hemocytometer.jpg" >}}

In the image, present in the foreground are the microalgae cells to be counted and the background, volumetric grids. 
This counting method is also known as direct counting.
Direct counting is the easiest counting method that can be performed by anyone without the need of sophisticated instruments.
If done properly and meticulously, this method can be accurate.

However, as with many other human-performed tasks, direct counting may be prone to human errors.
Can we instead offload the tedious and repetitive counting to a machine instead?
In this blog post, we showcase a proof-of-concept idea on using deep learning to count microalgae cells and a step-by-step walkthrough on building it.

This blog post we will cover the following:

+ How train a deep learning model with [IceVision](https://github.com/airctic/icevision) package.
+ Hosting the model on [Hugging Face Spaces](https://huggingface.co/spaces) for remote inferencing.
+ Deploy an app on Android with the [Flutter](https://flutter.dev/) framework.

Did I mention that all the tools used in the project are open-source and free of charge? Yes!! If you're ready let's begin.


### Model Development with IceVision
In this section we are going to start developing our deep learning model using [IceVision](https://github.com/airctic/icevision) - A computer vision framework package in Python.
We will be using Google Colab to develop our model. If you're ready, let's begin!
The notebook that we will use can be found here.

#### Installation
In order to use IceVision, we need to install it by running the following line.

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

### Conclusion
Github [repo](https://github.com/dnth/webdemo-microalgae-detection)

{{< figure_resizing src="microsense_logo.png" link="https://play.google.com/store/apps/details?id=com.micro.sense">}}
The app can be found in Google Playstore by the name MicroSense.