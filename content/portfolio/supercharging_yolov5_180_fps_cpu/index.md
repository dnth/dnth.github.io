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
After months of searching, you've finally found the *one*. 

The *one* object detection library that just works.
No installation hassle, no package version mismatch, and no `CUDA` errors. 

I'm taking about the ubiquitous [YOLOv5](https://github.com/ultralytics/yolov5) object detection library by [Ultralytics](https://ultralytics.com/yolov5).

Elated, you quickly find an interesting dataset from [Roboflow](https://roboflow.com/) and finally trained a state-of-the-art (SOTA) YOLOv5 model to detect firearms from image streams.

You ran through a quick checklist --
+ Inference results, checked ✅
+ COCO mAP, checked ✅
+ Live inference latency, checked ✅

You're on top of the world. 

You can finally pitch the results to your clients next Monday.
At the back of your mind, you imagined the clients' impressed looks on the astonishing feat.

Finally, on the pitching day, just when you thought things are going in the right direction.
One of the clients asked

"**Does your model run on our existing CPU?**"

You flinched. That wasn't something you anticipated. You tried to convince them that GPUs are the way forward and it is the "best" way to run your model in real-time.

You scanned the room and begin to notice their faces, they don't look particularly happy 👇

{{< figure_resizing src="meme.jpg">}}

I hope nobody will ever have to face this awkward situation in a pitching session, ever.
You don't have to learn it the hard way, like I did.

How, you asked? This post is exactly that.

I will show you how you can supercharge your YOLOv5 inference performance running on a consumer CPU.
Oh, we'll be using free and open-source tools from [Neural Magic](https://neuralmagic.com/). 

{{< notice tip >}}
By the end of this post, you will learn how to:

* Train state-of-the-art YOLOv5 model with your own data.
* Sparsify the model using SparseML quantization aware training loop.
* Export the sparsified model and run it using the DeepSparse engine at insane speed. 

**P/S**: The end result - YOLOv5 on CPU at 180+ FPS using only 4 cores! 🚀
{{< /notice >}}

If that sounds interesting let's get into it ⛷.


### 🔩 Setting Up


#### 🔫 Dataset

The [recent gun violence](https://edition.cnn.com/2022/05/25/us/uvalde-texas-elementary-school-shooting-what-we-know/index.html) news had me thinking deeply about how we can prevent incidents like these again. 
This is the worst gun violence since 2012, and 21 innocent lives were lost.

My heart goes out to all victims of the violence and their loved ones.

I'm not a lawmaker, so there is little I can do there. 
But, I think I know something in computer vision that might help.
That's when I came across the [Pistols Dataset](https://public.roboflow.com/object-detection/pistols) from Roboflow.


This dataset contains 2986 images and 3448 labels across a single annotation class: pistols. Images are wide-ranging: pistols in-hand, cartoons, and staged studio quality images of guns. The dataset was originally released by the University of Grenada.

{{< figure_resizing src="pistol.png">}}

#### 🦸 YOLOv5 Object Detection Library
Let's first install the YOLOv5 object detection library from Github.

For this post, we are going to use a [forked version](https://github.com/neuralmagic/yolov5) of the YOLOv5 library that will allow us to do custom optimizations in the upcoming section.

To install, run the following commands

```bash
git clone https://github.com/neuralmagic/yolov5.git
cd yolov5
git checkout release/0.12
pip install -r requirements.txt
```

Now let's put the downloaded Pistols Dataset into the appropriate folder for us to start training.
You can refer to my folder structure [here](https://github.com/dnth/yolov5-deepsparse-blogpost).
Feel free to fork and use it on your own dataset.

#### 🥋 Training

To start training we will run the `train.py` script from the YOLOv5 repo.

```bash
python train.py --cfg ./models_v5.0/yolov5s.yaml \
                --data pistols.yaml \
                --hyp data/hyps/hyp.scratch.yaml \
                --weights yolov5s.pt --img 416 --batch-size 64 \
                --optimizer SGD --epochs 240 --device 0 \
                --project yolov5-deepsparse --name yolov5s-sgd
```


### ⛳ Baseline Inference
Let's first establish a baseline before we start optimizing.

#### 🔦 PyTorch

Inference on CPU with YOLOv5-S PyTorch model.

On a Intel i9-11900 8 core processor

+ Average FPS : 21.91
+ Average inference time (ms) : 45.58

{{< video src="vids/torch-annotation/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

On a RTX3090 GPU

+ Average FPS : 89.20
+ Average inference time (ms) : 11.21

{{< video src="vids/torch-gpu/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}


#### 🕸 DeepSparse Engine
Out of the box, no modifications to the model running at 4 CPU cores.
Input the unoptimized onnx model.

+ Average FPS : 29.48
+ Average inference time (ms) : 33.91

{{< video src="vids/onnx-annotation/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}


### 🌀 Sparsify with SparseML
Sparsification is the process of removing redundant information from a model.

[SparseML](https://github.com/neuralmagic/sparseml) is an open-source library by Neural Magic to apply sparsification recipes to neural networks.
It currently supports integration with several well known libraries from computer vision and natural language processing domain.


Sparsification results in a smaller and faster model. 
This is how we can significantly speed up our YOLOv5 model, by a lot!

There are several ways to sparsify models with SparseML:
+ Post-training (One-shot) - Quantization
+ Training Aware - Pruning & Quantization
+ Sparse Transfer Learning

#### ☝️ One-Shot
No re-training. Just dynamic quantization. Easiest.

+ Average FPS : 32.00
+ Average inference time (ms) : 31.24

{{< video src="vids/one-shot/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

#### ✂ Pruned YOLOv5-S
Re-training with recipe.

+ Average FPS : 35.50
+ Average inference time (ms) : 31.73

{{< video src="vids/yolov5s-pruned/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

#### 🪚 Pruned + Quantized YOLOv5-S
Re-training with recipe.

+ Average FPS : 58.06
+ Average inference time (ms) : 17.22

{{< video src="vids/yolov5s-pruned-quant/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}


#### 🤹‍♂️ Sparse Transfer Learning
Taking an already sparsified (pruned and quantized) and fine-tune it on your own dataset.

+ Average FPS : 51.56
+ Average inference time (ms) : 19.39
{{< video src="vids/yolov5s-pruned-quant-tl/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

### 🚀 Supercharging FPS

Pruned and Quantized YOLOv5n + Hardswish Activation
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