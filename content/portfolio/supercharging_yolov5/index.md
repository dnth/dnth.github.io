---
title: "Supercharging YOLOv5: How I get 180+ FPS Inference on CPUs with only 4 Cores"
date: 2022-01-19T11:00:15+08:00
featureImage: images/portfolio/supercharging_yolov5/thumbnail.gif
postImage: images/portfolio/supercharging_yolov5/post_image.png
tags: ["DeepSparse", "ONNX", "YOLOv5", "real-time", "optimization", "pistol"]
categories: ["deployment", "object-detection", "modeling"]
toc: true
socialshare: true
description: "Accelerate inference up to 70 FPS on a CPU!"
images : 
- images/portfolio/supercharging_yolov5/post_image.png
---

{{< notice info >}}
This blog post is still a work in progress. If you require further clarifications before the contents are finalized, please get in touch with me [here](https://dicksonneoh.com/contact/), on [LinkedIn](https://www.linkedin.com/in/dickson-neoh/), or [Twitter](https://twitter.com/dicksonneoh7).
{{< /notice >}}

### 🔥 Motivation
So you've trained the most popular object detection model - YOLOv5 and got some impressive results.

### ⛳ Baseline YOLOv5 inference

Inference on CPU with YOLOv5-S PyTorch model.

+ Average FPS : 21.91
+ Average inference time (ms) : 45.58

{{< video src="vids/torch-annotation/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}


### 🪄 DeepSparse Engine
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

### ✂ Pruned YOLOv5

+ Average FPS : 35.50
+ Average inference time (ms) : 31.73
{{< video src="vids/yolov5s-pruned/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

### ♒ Pruned and Quantized YOLOv5

+ Average FPS : 58.06
+ Average inference time (ms) : 17.22
{{< video src="vids/yolov5s-pruned-quant/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}


### 🤹‍♂️ Sparse Transfer Learning
+ Average FPS : 51.56
+ Average inference time (ms) : 19.39
{{< video src="vids/yolov5s-pruned-quant-tl/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

### 🚀 Pruned and Quantized YOLOv5n + Hardswish

+ Average FPS : 93.33
+ Average inference time (ms) : 10.71
{{< video src="vids/yolov5n-pruned-quant/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

### 🚧 Conclusion



### 🙏 Comments & Feedback
I hope you've learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter post or [drop me a message](https://dicksonneoh.com/contact/).
{{< tweet dicksonneoh7 1527512946603020288>}}


If you like what you see and don't want to miss any of my future content, follow me on Twitter and LinkedIn where I deliver more of these tips in bite-size posts.