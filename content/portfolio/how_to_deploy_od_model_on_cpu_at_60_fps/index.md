---
title: "[WIP] Deploy Object Detection Models on CPU at 50+ FPS"
date: 2022-04-29T15:00:15+08:00
featureImage: images/portfolio/how_to_deploy_od_model_on_cpu_at_60_fps/thumbnail.gif
postImage: images/portfolio/how_to_deploy_od_model_on_cpu_at_60_fps/post_image.png
tags: ["OpenVINO", "YOLOX", "ONNX", "CVAT"]
categories: ["deployment", "object-detection"]
toc: true
socialshare: true
description: "10x your YOLOX model in few simple steps!"
images : 
- images/portfolio/how_to_deploy_od_model_on_cpu_at_60_fps/post_image.png
---

{{< notice info >}}
This blog post is still a work in progress. If you require further clarifications before the contents are finalized, please get in touch with me [here](https://dicksonneoh.com/contact/), on [LinkedIn](https://www.linkedin.com/in/dickson-neoh/), or [Twitter](https://twitter.com/dicksonneoh7).
{{< /notice >}}

### 🚦 Motivation
We train on GPU. CPU is more common for deployment.
OD models run very slow on CPU. Lightest model runs less than 10 fps.
Danger for mission critical application such as self-driving car that require real-time responses.

{{< video src="yolox.mp4" width="700px" loop="true" autoplay="true" >}}

{{< notice tip >}}
By the end of this post, you will learn about:

* Train state-of-the-art YOLOX model with your own data.
* Convert the .pth model into ONNX and IR model.
* Run post training quantization to 10x model inference speed.
{{< /notice >}}


### ⛷ Modeling with YOLOX
Labeling on CVAT
Training with YOLOX model.
Anchor free.
Runs fast.

{{< figure_resizing src="cvat_new.png" caption="Create a new task on CVAT.">}}

{{< figure_resizing src="task_description.png" caption="Input the images and labels.">}}

{{< figure_resizing src="draw_box.png" caption="Draw.">}}

{{< figure_resizing src="show_bbox.png" caption="Show.">}}

{{< figure_resizing src="export.png" caption="Export.">}}

{{< figure_resizing src="coco.png" caption="COCO format.">}}

### 🤖 ONNX Runtime
```bash
pip install onnxruntime
```

```bash
python src/export_onnx.py --output-name models/ONNX/yolox_s_lp.onnx -f exps/YOLOX_S/yolox_s_lp.py -c YOLOX_outputs/yolox_s_lp/best_ckpt.pth
```

### 🔗 OpenVINO Intermediate Representation

```bash
pip install openvino-dev[onnx]==2022.1.0
```

```bash
mo --input_model models/ONNX/yolox_s_lp.onnx --input_shape [1,3,640,640] --data_type FP16 --output_dir models/IR/
```

### 🛠 Post-Training Optimization Toolkit
OpenVINO provides a higly under-rate Post-training Optimization Toolkit (POT).
Runs algorithm for 8-bit quantization over a wide variety of DNN models.

From the OpenVINO documantation [page](https://docs.openvino.ai/2021.1/pot_compression_algorithms_quantization_README.html): 

+ `DefaultQuantization` is a default method that provides fast and in most cases accurate results for 8-bit quantization. 

+ `AccuracyAwareQuantization` enables remaining at a predefined range of accuracy drop after quantization at the cost of performance improvement. It may require more time for quantization.


```bash
pot -q default -m models/IR/yolox_s_lp.xml -w models/IR/yolox_s_lp.bin --engine simplified --data-source data/pot_images --output-dir models/INT8
```

### 🚀 Real-time Inference @50+ FPS
The moment of truth
{{< video src="int8.mp4" width="700px" loop="true" autoplay="true" >}}

### 🏁 Conclusion
In this post you've learned how to 10x your object detection model using the techniques covered.

### 🙏 Comments & Feedback
If you like this and don't want to miss any of my future content, follow me on [Twitter](https://twitter.com/dicksonneoh7) and [LinkedIn](https://www.linkedin.com/in/dickson-neoh/) where I share more of these in bite-size posts.

If you have any questions, comments, or feedback, please leave them on the following Twitter post or [drop me a message](https://dicksonneoh.com/contact/).
{{< tweet 1517004585495240704>}}





