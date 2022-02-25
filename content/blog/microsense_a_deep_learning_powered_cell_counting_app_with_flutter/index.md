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

Did I mention that all the tools used in this project are open-source and free of charge? Yes!! If you're ready let's begin.


### Model Development with IceVision
In this section we are going to start developing our deep learning model using IceVision with a custom dataset.
You can either choose to develop the model on your local machine or in the cloud.
For simplicity, we will be using Google Colab to develop this model.
[Here](https://colab.research.google.com/github/dnth/dnth.github.io/blob/main/content/blog/microsense_a_deep_learning_powered_cell_counting_app_with_flutter/training_vfnet.ipynb) is an accompanying Colab notebook if you wish to follow along.


#### Preparing the dataset
Before embarking on any machine learning work, we must ensure that we have a dataset to work on.
Our task at hand is to construct a model that can count microalgaes. 
Since there are no public dataset available, we will have to curate our own dataset.
The image below shows a dozen images take with a hemocytometer from a water sample belonging to a pond in Nilai, Malaysia.
{{< figure_resizing src="dataset_sample.png" >}}

There is only one issue now, and that is the images are not annotated. We will have to annotate all the images with an open source image labeling tool known as [labelImg](https://github.com/tzutalin/labelImg).
`labelImg` allows us to annotate any images with class labels and bounding boxes surrounding the object of interest.
The following figure shows a demo of `labelImg` taken from the GitHub repository.
{{< figure_resizing src="labelimg_demo.jpg" >}}

The easiest way to install `labelImg` on your local machine is to run `pip3 install labelImg` in the terminal.
Once done, type `labelImg` in the same terminal and a window should pop open as shown in the image below.

{{< figure_resizing src="labelimg_start.png" >}}

Now, let's load the folder that contains the microalgae images into `labelImg` and annotate them! To do that, click on the **Open Dir** icon and navigate to the folder containing the images. An image should now show up in `labelImg`.
Next click on the **Create RectBox** icon to start drawing bounding boxes around the microalgaes. Next you will be prompted to enter a label name. 
Key in microalgae as the label name. Once done, a rectangular bounding box should appear on-screen.

{{< figure_resizing src="labelimg_loaded.png" >}}

Now comes the repetitive part, we will need to draw a bounding box for each microalgae cell for all images in the folder.
To accelerate the process I highly recommend the use of Hotkeys keys with `labelImg`.
{{< figure_resizing src="hotkeys.png" width=400 >}}

Once done, remember to save the annotations. The annotations are saved in `XML` file with a filename matching to image as shown below.
{{< figure_resizing src="xml_files.png" >}}

Once all images are labelled, we will partition the image and annotations into three sets namely train set, validation set and test set.
These will be used to train and evaluate our model in the next section.

It took a few hours to meticulously label the images. The labeled images can be downloaded here.

#### Training
IceVision computer vision framework.

#### Local Inference
Inference on a local machine
{{< figure_resizing src="inference.png" >}}

Figure illustrates the raw detection of cells from microscope image. The model is a RetinaNet with a ResNet50 backbone trained using [IceVision](https://github.com/airctic/icevision).
{{< figure_resizing src="detection.png" >}}

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