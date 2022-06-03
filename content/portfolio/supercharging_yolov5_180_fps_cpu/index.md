---
title: "Supercharging YOLOv5: How I get 180+ FPS Inference on a CPU using only 4 Cores"
date: 2022-01-19T11:00:15+08:00
featureImage: images/portfolio/supercharging_yolov5/thumbnail.gif
postImage: images/portfolio/supercharging_yolov5/post_image.png
tags: ["DeepSparse", "ONNX", "YOLOv5", "real-time", "optimization", "pistol"]
categories: ["deployment", "object-detection", "modeling"]
toc: true
socialshare: true
description: "Accelerate inference up to 180 FPS on a CPU!"
images : 
- images/portfolio/supercharging_yolov5/post_image.png
---

{{< notice info >}}
This blog post is still a work in progress. If you require further clarifications before the contents are finalized, please get in touch with me [here](https://dicksonneoh.com/contact/), on [LinkedIn](https://www.linkedin.com/in/dickson-neoh/), or [Twitter](https://twitter.com/dicksonneoh7).
{{< /notice >}}

### 🔥 Motivation
So you've trained the most popular object detection model - YOLOv5 and got some impressive results.

Now you finally get to deploy and run the model on a real.

But there's only one problem. Not everyone has a GPU to run the model.

{{< figure_resizing src="meme.jpg">}}

I will show you how you can supercharge your YOLOv5 inference performance from 20+ FPS to 180+ FPS on a CPU.
With free and open-source tools from [Neural Magic](https://neuralmagic.com/) you'll get GPU-class performance on commodity CPUs.





{{< notice tip >}}
By the end of this post, you will learn how to:

* Train state-of-the-art YOLOv5 model with your own data.
* Prune and quantize the YOLOv5 model using SparseML.
* Export the sparsified YOLOv5 and run it using the DeepSparse engine. 

**P/S**: You'll get GPU-class inference performance on a CPU using only 4 cores! 😱
{{< /notice >}}

### 🕸 Sparse Neural Network


### 🔩 Setting Up

Use Ultralytic YOLOv5 implementations.
Clone repo from NM fork.

Roboflow pistol dataset.

### ⛳ Baseline PyTorch Inference

Inference on CPU with YOLOv5-S PyTorch model.

On a Intel i9-11900 8 core processor

+ Average FPS : 21.91
+ Average inference time (ms) : 45.58

{{< video src="vids/torch-annotation/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

On a RTX3090 GPU

+ Average FPS : 89.20
+ Average inference time (ms) : 11.21

{{< video src="vids/torch-gpu/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}


### 🪄 Dense Model Inference with DeepSparse Engine
Out of the box, no modifications to the model.
Input the unoptimized onnx model.

+ Average FPS : 29.48
+ Average inference time (ms) : 33.91

{{< video src="vids/onnx-annotation/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}


### 🌀 Sparsify Models with SparseML
Sparsification is the process of removing redundant information from a model.

Several ways:
+ One shot
+ Training Aware
+ Transfer Learning

### ☝️ One Shot
No re-training. Just dynamic quantization. Easiest.

+ Average FPS : 32.00
+ Average inference time (ms) : 31.24

{{< video src="vids/one-shot/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

### ✂ Pruned YOLOv5
Re-training with recipe.

+ Average FPS : 35.50
+ Average inference time (ms) : 31.73

{{< video src="vids/yolov5s-pruned/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

### ♒ Pruned and Quantized YOLOv5
Re-training with recipe.

+ Average FPS : 58.06
+ Average inference time (ms) : 17.22

{{< video src="vids/yolov5s-pruned-quant/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}


### 🤹‍♂️ Sparse Transfer Learning
Taking an already sparsified (pruned and quantized) and fine-tune it on your own dataset.

+ Average FPS : 51.56
+ Average inference time (ms) : 19.39
{{< video src="vids/yolov5s-pruned-quant-tl/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

### 🚀 Pruned and Quantized YOLOv5n + Hardswish Activation
Hardswish activation performs better with DeepSparse.


+ Average FPS : 93.33
+ Average inference time (ms) : 10.71
{{< video src="vids/yolov5n-pruned-quant/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

### 🚧 Conclusion
In this blog post I've shown you


### 🙏 Comments & Feedback
I hope you've learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter post or [drop me a message](https://dicksonneoh.com/contact/).
{{< tweet dicksonneoh7 1527512946603020288>}}


If you like what you see and don't want to miss any of my future content, follow me on Twitter and LinkedIn where I deliver more of these tips in bite-size posts.