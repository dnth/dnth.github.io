---
title: "[WIP] Deploying Any Object Detection Model on Mobile Devices with Flutter for Busy Engineers"
date: 2022-04-14T15:07:15+08:00
featureImage: images/portfolio/deploying_od_models_flutter_busy_enginners/thumbnail.gif
postImage: images/portfolio/deploying_od_models_flutter_busy_enginners/post_image.png
tags: ["Flutter", "Hugging-Face", "Android", "Fast.ai", "Gradio", "IceVision"]
categories: ["deployment", "object-detection"]
toc: true
socialshare: true
description: "Leveraging giant models on the clouds in your palms."
images : 
- images/portfolio/deploying_od_models_flutter_busy_enginners/post_image.png
---

{{< notice info >}}
This blog post is still a work in progress. If you require further clarifications before the contents are finalized, please get in touch with me [here](https://dicksonneoh.com/contact/), on [LinkedIn](https://www.linkedin.com/in/dickson-neoh/), or [Twitter](https://twitter.com/dicksonneoh7).
{{< /notice >}}

### Deployment - where most models fail
According to [Gartner](https://www.gartner.com/en/newsroom/press-releases/2018-02-13-gartner-says-nearly-half-of-cios-are-planning-to-deploy-artificial-intelligence), more than 85% of machine learning models never made it into production.
This trend is expected to continue further this year in 2022. 

Despite huge efforts in research, machine learning models often fail in the deployment.

What exactly is deployment? 
In simple terms, deployment means making a model's capability or insight available to users or other systems - [Luigi Patruno](https://mlinproduction.com/what-does-it-mean-to-deploy-a-machine-learning-model-deployment-series-01/).
Only when a model is deployed can they add value to businesses or organizations 

In this post, I will outline basic steps to deploy models on lightweight mobile devices running on Android operating system.

By the end of this post you will learn about

* What are deployment patterns.
* How to host a model on Hugging Face.
* Deploying on with HTTP calls.
* Displaying the results on a Flutter Android app.

### Deployment architecture
Deploying deep learning model on the edge is not trivial.
Models can be huge, complex and requires a lot of resources to run.

In order to keep the Android app lightweight, we kept the model in a cloud inference server where it can serve the prediction to our mobile app via API calls.
At the same time, we can also monitor and update the model in the inference server by deploying them from our local machine used for training.
{{< figure_resizing src="architecture.png" caption="Deployment architecture.">}}


### Hosting model on Hugging Face

Assume you already have a model ready.
For this post I will use a trained IceVision model.

Assume you already have a trained model.
Publish model on Hugging Face Space with Gradio.
Use Gradio to expose the model HTTP endpoint.

https://hf.space/embed/dnth/webdemo-microalgae-counting/+

### Calling HTTP Endpoint in Flutter
What if Flutter.
Using Flutter to call the HTTP endpoint.
Send images, get predictions.

{{< figure_resizing src="api_endpoint.png" caption="Exposed endpoint.">}}

### Displaying results
Decode prediction.
The screenshot below illustrates the Android app sending a sample image to the inference server and getting a response on the number of detected microalgae cells on the image.
{{< figure src="microsense.gif" width=500 >}}

### Up Next
Hosting on AWS Lambda.
Using Hugging Face Inference API.

<!-- ### Motivation
Deploying 


### Architecture
The image below illustrates the architecture of this work.




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


The app can be found in Google Playstore by the name [MicroSense]((https://play.google.com/store/apps/details?id=com.micro.sense)).
{{< figure_resizing src="microsense_logo.png" link="https://play.google.com/store/apps/details?id=com.micro.sense">}}
Try it out and leave us a message if you find it useful or are keen to develop the app further. 
If you're interested to find out how I made the app from scratch, I wrote a tutorial blog post about it [here](https://dicksonneoh.com/blog/deep_learning_on_android_with_flutter_and_hugging_face/). -->
