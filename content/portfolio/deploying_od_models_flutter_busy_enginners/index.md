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

### 🚑 Deployment: ML Models Graveyard
According to [Gartner](https://www.gartner.com/en/newsroom/press-releases/2018-02-13-gartner-says-nearly-half-of-cios-are-planning-to-deploy-artificial-intelligence), more than **85%** of machine learning (ML) models never made it into production.
This trend is expected to continue further this year in 2022.

In other words, despite all the promises and hype around ML, most models fails to deliver in a production environment.
According to Barr Moses, CEO, Monte Carlo, [deployment](https://towardsdatascience.com/why-production-machine-learning-fails-and-how-to-fix-it-b59616184604) is one of the critical points where many models fail.

So what exactly is deployment of ML model? 
Simply put, deployment is making a model's capability or insight available to other users or systems - [Luigi Patruno](https://mlinproduction.com/what-does-it-mean-to-deploy-a-machine-learning-model-deployment-series-01/).

**Only when a model is deployed can they add value to businesses or organizations.**

In this post, I will outline the basic ideas to deploy ML models lightweight mobile devices **easily, quickly, for free**.
By the end of this post, you will learn about:

* Setting up interactive demo for rapid feedback and iteration.
* How to host a model on Hugging Face.
* Deploying on any edge device with HTTP calls.
* Displaying the results on a Flutter Android app.


### 🪜 Begin with deployment in mind
Many ML researchers take pride in training bleeding edge models with state-of-the-art (SOTA) results on datasets.
As a research scientist myself, I understand how deeply satisfying it gets training them successfully.

Unfortunately, many of these so called "SOTA models" will just end up on preprints, (or jupyter notebook) or in some obscure repository nobody cares about after the initial hype. 

Eventually, they are forgotten and lost in the ocean of newer "SOTA models".
To make things worse, the obsession of chasing after "SOTA models" often cause researchers to lose track of the end goal of building the model itself - deployment.

<!-- We can forget about ever finding these models in production. -->

{{< figure_resizing src="jupyter_meme.png" caption="Source: ClearML on Reddit." link="https://www.reddit.com/r/mlops/comments/o8w2e4/you_know_the_deal_if_you_dont_post_content_ill/?utm_source=share&utm_medium=ios_app&utm_name=iossmf">}}

To mitigate this, it is helpful if we build models with deployment in mind, as the end result.
This is the beginning to getting a model into production. 

Deployment, is unfortunately a messy and complicated topic in MLOps for us deeply dive in here. That is not the purpose of this post.

**My objective in this post is to show you how you can deploy an ML model easily on a mobile device without getting your hands dirty with servers, backend or Kubernetes.**

<!-- Once the model is built, we can immediately spin up an interactive demo.
An interactive demo opens the door to users' feedbacks from using the model which are invaluable in product iteration to prepare for further stages.

Unfortunately, many don't even make it through this phase.
This is not anyone's fault, as making an interactive demo often requires skills beyond ML. -->

<!-- In this post I'm going to show you that is no longer the case.

**Anyone with no knowledge about backend, servers, or Kubernetes can quickly spin up an interactive demo, deploy them on the cloud or on a mobile device and share it to users to gain feedback.** -->

The following figure shows the deployment architecture that allows us to accomplish that.
{{< figure_resizing src="architecture.png" caption="Deployment architecture.">}}

### 🤗 Hosting a model on Hugging Face
The first part of the puzzle is to host our model on some cloud infrastructure.
In this post, let's use a free service known as Hugging Face *Spaces*.

*Spaces* is a platform where anyone can upload their model and share to the world.
If you head to https://huggingface.co/spaces, you will find thousands of models that researchers made freely available online.
{{< figure_resizing src="spaces_web.png">}}

These models are hosted on *Spaces* for demo and sharing purposes. 
But they can be scaled up into full fledge production with the [Inference API](https://huggingface.co/inference-api).

Let's set up a Space to host our model. If you're unsure how to do that, I wrote a recent guide on how to set your own Space with the Gradio app [here](https://dicksonneoh.com/portfolio/deploy_icevision_models_on_huggingface_spaces/).

In this post, I will use an IceVision object detection model trained to detect microalgae cells from image.
I trained this model in under a minute with 17 labeled images. [Here's how I did it]((https://dicksonneoh.com/portfolio/training_dl_model_for_cell_counting/)).

Once the Space is set, we will have a Gradio interface like the following
{{< figure_resizing src="space_demo.png" caption="Screenshot from Hugging Face Space.">}}

This Space is now ready to be shared to anyone with an internet connection and a browser.
Click [here](https://hf.space/embed/dnth/webdemo-microalgae-counting/+) if you'd like to check out it out.

But what if we want to make the app work on a mobile device **without using a browser?** Enter 👇

### 🔧 Calling the HTTP Endpoint
One neat feature of the Gradio app is it exposes the model through a RESTful API.
This makes the model prediction accessible via HTTP request which we can conveniently use on any mobile device!

Now, any computationally lightweight device can make use of the model's prediction just by running a simple HTTP call.
All the heavy lifting is taken care by the Hugging Face Spaces infrastructure. 

**This can be a game-changer if the model is complex and the edge device is not powerful enough to run the model - which is a common scenario.**

Additionally, this also reduces deployment hardware cost, because now any lightweight mobile device can leverage the model's capability.

The figure below shows the endpoint for us to call the model.

{{< figure_resizing src="api_endpoint.png">}}

As shown, the input to the model is an image and the output, and image (with bounding boxes) and also a value of the microalgae count. You can check out the API [here](https://hf.space/embed/dnth/webdemo-microalgae-counting/api).

Let's copy the `URL endpoint` and use in the next section



### 📲 Displaying results in Flutter
Decode prediction.
The screenshot below illustrates the Android app sending a sample image to the inference server and getting a response on the number of detected microalgae cells on the image.
{{< figure src="microsense.gif" width=500 >}}

### 💡 Up Next
Scaling up.
Hosting on AWS Lambda.
Using Hugging Face Inference API.


### 🙏 Comments & Feedback
If you like this and don't want to miss any of my future posts, follow me on [Twitter](https://twitter.com/dicksonneoh7) and [LinkedIn](https://www.linkedin.com/in/dickson-neoh/) where I share more of these contents in a bite size post.

If you have any questions, comments, or feedback, please you can leave them on the following Twitter post or [drop me a message](https://dicksonneoh.com/contact/).
<!-- {{< tweet 1513478343726809090>}} -->

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