---
title: "MicroSense - A Deep Learning Android App for Cell Counting"
date: 2022-02-24T15:07:15+08:00
thumbnail: images/portfolio/microsense_a_deep_learning_powered_cell_counting_app_with_flutter/feature_image.gif
postImage: images/portfolio/microsense_a_deep_learning_powered_cell_counting_app_with_flutter/post_image.jpg
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
In this we we showcase a proof-of-concept app.


### Flutter App (WIP)
Github [repo](https://github.com/dnth/webdemo-microalgae-detection)

{{< figure_resizing src="microsense_logo.png" link="https://play.google.com/store/apps/details?id=com.micro.sense">}}
The app can be found in Google Playstore by the name MicroSense.


Figure shows an Android app written using the Flutter framework. The inference is done by calling the API from our Space.
Checkout the app published on Google Playstore [here](https://play.google.com/store/apps/details?id=com.micro.sense).
{{< figure src="microsense.gif" >}}