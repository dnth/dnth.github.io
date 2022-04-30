---
title: "[WIP] How to deploy object detection models on CPU at 60 FPS"
date: 2022-04-29T15:00:15+08:00
featureImage: images/portfolio/how_to_deploy_od_model_on_cpu_at_60_fps/thumbnail.gif
postImage: images/portfolio/how_to_deploy_od_model_on_cpu_at_60_fps/post_image.png
tags: ["OpenVINO", "YOLOX", "ONNX", "CVAT"]
categories: ["deployment", "object-detection"]
toc: true
socialshare: true
description: "Real-time detection on the edge at 60FPS."
images : 
- images/portfolio/how_to_deploy_od_model_on_cpu_at_60_fps/post_image.png
---

{{< notice info >}}
This blog post is still a work in progress. If you require further clarifications before the contents are finalized, please get in touch with me [here](https://dicksonneoh.com/contact/), on [LinkedIn](https://www.linkedin.com/in/dickson-neoh/), or [Twitter](https://twitter.com/dicksonneoh7).
{{< /notice >}}

### Motivation
We train on GPU. CPU is more common for deployment.
OD models run very slow on CPU. Lightest model runs less than 10 fps.
Danger for mission critical application such as self-driving car that require real-time responses.

### Modeling with YOLOX
Labeling on CVAT
Training with YOLOX model.
Anchor free.
Runs fast.

### ONNX Runtime

### OpenVINO Intermediate Representation

### Post Training Quantization

### Real-time Inference

### Conclusion

### Comments and Feedback

