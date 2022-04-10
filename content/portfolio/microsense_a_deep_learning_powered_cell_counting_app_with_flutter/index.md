---
title: "MicroSense - A Deep Learning Android App for Cell Counting"
date: 2022-02-24T15:07:15+08:00
featureImage: images/portfolio/microsense_a_deep_learning_powered_cell_counting_app_with_flutter/feature_image.gif
postImage: images/portfolio/microsense_a_deep_learning_powered_cell_counting_app_with_flutter/post_image.jpg
tags: ["Flutter", "HuggingFace", "Android"]
categories: ["MVP", "portfolio"]
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

In the image above, we find the microalgae cells in the foreground and the volumetric grids in the background. 
This counting method is also known as direct counting.
Direct counting is the easiest counting method that can be performed by anyone without the need of sophisticated instruments.
If done properly and meticulously, this method can be accurate.

However, as with many other human-performed tasks, direct counting may be prone to human errors.
Can we instead offload the tedious and repetitive counting to a machine instead?
In work, we showcase a minimum viable product (MVP) Android app that uses a deep learning object detection model in the background to count the microalgae cells.

### Architecture
The image below illustrates the architecture of this work.
In order to keep the Android app lightweight, we kept the model in a cloud inference server where it can serve the prediction to our mobile app via API calls.
At the same time, we can also monitor and update the model in the inference server by deploying them from our local machine used for training.

{{< figure_resizing src="architecture.png" >}}

### Speed-Accuracy Trade Off
One of the many concerns in putting a sophisticated deep learning model on an Android app is the portability.
Depending on the type of models, the size may range from few MB to a few hundred MBs.
This may sound trivial with cheap memory cost nowadays, but a mobile app with few hundreds of MBs in size will surely be impractical to keep on a device for long.
There are methods to reduce the size and computation of these models making them more mobile-friendly such as model pruning and quantization.
These however comes at the cost of reducing the accuracy and effectiveness of the model.
We are back to the classic speed vs accuracy trade off when it comes to model deployment on mobile devices.
On the one hand, we want the model to be as accurate and effective as possible, on the other hand we also need to make sure the model can be feasible run on lightweight mobile processors.
On some applications, we can certainly trade model accuracy for a huge gain in portability on mobile.
However, on many mission-critical applications, even a small reduction in model effectiveness could severely impact the outcome of the app.
This can be mitigated by using a remote infrastructure to host the model and leave the app lightweight, possibly gaining the best of both.

### Remote Inference
These limitations can be solved by offloading the heavy lifting of hosting and running the model to a remote server or the cloud.
There are many cloud based solutions that can perform the job, however in this example we utilized Hugging Face Space as a server for inferencing with [millisecond latency](https://huggingface.co/blog/infinity-cpu-performance).
The free tier offers up to 8 CPU cores and 16GB RAM making it extremely feasible to host almost any deep learning model available today at no cost.
The paid tier offers much more in terms of CPU performance, latency and throughput. They claim to accelerate the latency of Transformer based models to a [1ms latency](https://huggingface.co/infinity).

One neat feature available on Hugging Face Space is the built-in integration with Gradio.
Gradio is a user-interface app that makes it easy to deploy model demos on Hugging Face. 
Additionally, Gradio also exposes the model inference with a REST-ful API calls.
In other words, the model hosted on Hugging Face with Gradio can communicate with any app on the internet using standard HTTP calls.

In this MVP we hosted the object detection model on Hugging Face using the free tier.
This takes a huge burden off the Android app and significantly improves app portability without sacrificing model accuracy and effectiveness.
The drawback in this case is the latency of the model inference now depends on the network connection to the inference server which is not an issue for this app since we do not need a real-time inference.
However, with the 1ms latency claim on the paid tier, we wonder if real-time inference is possible. This is something we have not explored in this MVP. But it will be interesting to know.




### Android App
The Android app was built using the Google [Flutter](https://flutter.dev/) framework.
Now, instead of having to embed the model in the app, all we need to do is to send an `HTTP` `POST` request to the Hugging Face server with an image as the input.
Once the server receives the request, an inference on the model is run and the output is returned to the Android app as a response.
The screenshot below illustrates the Android app sending a sample image to the inference server and getting a response on the number of detected microalgae cells on the image.
{{< figure src="microsense.gif" >}}

The app can be found in Google Playstore by the name [MicroSense]((https://play.google.com/store/apps/details?id=com.micro.sense)).
{{< figure_resizing src="microsense_logo.png" link="https://play.google.com/store/apps/details?id=com.micro.sense">}}
Try it out and leave us a message if you find it useful or are keen to develop the app further. 
If you're interested to find out how I made the app from scratch, I wrote a tutorial blog post about it [here](https://dicksonneoh.com/blog/deep_learning_on_android_with_flutter_and_hugging_face/).
