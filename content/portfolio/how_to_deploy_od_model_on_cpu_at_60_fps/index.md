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

{{< notice tip >}}
By the end of this post, you will how to:

* Train state-of-the-art YOLOX model with your own data.
* Convert the YOLOX PyTorch model into ONNX and OpenVINO IR format.
* Run quantization algorithm to 10x your model's inference speed. 

**P/S**: The final model runs faster on the CPU than the GPU! 😱
{{< /notice >}}


Deep learning (DL), they seem to be in popular demand these days. 
We find them everywhere - in blog posts, articles, research papers, Jupyter notebooks. 
Except in production 🤷‍♂️.

As much as we want to believe DL can solve our machine learning problems, more than 85% of DL models don't make it into production - according to a recent survey by [Gartner](https://www.gartner.com/en/newsroom/press-releases/2018-02-13-gartner-says-nearly-half-of-cios-are-planning-to-deploy-artificial-intelligence).

The barrier? *Deployment*.

For object detection models, we typically train them on massive GPUs either locally or in the cloud.
But when it comes to deploying the model, running them on GPUs is impractical in most situations.

On the other hand, CPUs are far more common in deployment settings, and a lot cheaper too.
But DL models are orders of magnitude slower when deployed on CPUs right?

No. This is no longer true.

In this post, I will walk you through how we go from this 🐌

{{< video src="yolox_cpu.mp4" width="700px" loop="true" autoplay="true">}}

to this 🚀
{{< video src="int8.mp4" width="700px" loop="true" autoplay="true">}}


Yes you read that right, the model runs on a CPU 😱.
If that looks interesting, let's dive in.


### ⛷ Modeling with YOLOX
{{< figure_resizing src="yolox_demo.png">}}

We will use the state-of-the-art YOLOX model to detect the license plate of vehicles around the neighborhood.

Before we can do that, we will need to collect images of the license pl ate and annotate them.
I collected about 40 images in total. 30 will be used as the training set and 10 as the validation set.

To label them, let's use the open-source CVAT labeling tool by Intel.
There are a ton of other labeling tools out there feel free to use them if you are comfortable.

If you'd like to try CVAT, head to https://cvat.org/ and create an account and log in.

As shown below, click on *Task*, fill in the approriate details like name of the task, add related labels and upload the images.
{{< figure_resizing src="cvat_new.png" caption="Create a new task on CVAT.">}}

Since we are detecting the license plate, I've entered only one label - LP (license plate).
Once completed, you will see a summary page as below. Click on `Job #368378` at ③ and it should bring you to the labeling page.
{{< figure_resizing src="task_description.png" caption="Input the images and labels.">}}

To start labeling, click on the square icon at ① and click Shape at ② in the figure below.
{{< figure_resizing src="draw_box.png" caption="Draw.">}}

You can then start drawing bounding boxes around the license plate. Do this for all 40 images.
{{< figure_resizing src="show_bbox.png" caption="Show.">}}

Once done, we are ready to export the annotations on the *Tasks* page.
{{< figure_resizing src="export.png" caption="Export.">}}

Make sure the format is *COCO 1.0* and click on OK.
{{< figure_resizing src="coco.png" caption="COCO format.">}}


Now, we have our dataset ready let's train our YOLOX model.
If you havent already install the YOLOX library by following the instructions [here](https://github.com/Megvii-BaseDetection/YOLOX).

Place your images and annotations in the `datasets` folder following the structure outlined [here](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/datasets).

You will also need to prepare a few the `Exp` config file before starting the training loop. They are documented [here](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/docs/train_custom_data.md). 
The `Exp` file lets you configure everything about the model from the location of the dataset, model architecture, and training hyperparameters.

For simplicity let's use the `YOLOX-s` model. There are a host of other YOLOX models you can try like `YOLOX-m`, `YOLOX-l`, `YOLOX-x`, `YOLOX-Nano`, `YOLOX-Tiny`, etc. The details are on the README on the YOLOX repo. Feel free to experiment.

My `Exp` file looks like the following

```python {linenos=table}
import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "data"
        self.train_ann = "train.json"
        self.val_ann = "val.json"

        self.num_classes = 1
        self.data_num_workers = 4
        self.eval_interval = 1

        # --------------- transform config ----------------- #
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mosaic_scale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
```

Once done, we can run the provided training script in the `tools` folder 
```bash
python tools/train.py -f exps/example/custom/yolox_s.py -d 8 -b 64 --fp16 -o -c /path/to/yolox_s.pth
```

After the training completes, you can use the model to run inference by using the `demo.py` script in the same folder.

```bash
python tools/demo.py video -n yolox-s -c /path/to/your/yolox_s.pth --path /path/to/your/video --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```

I'm running this on my machine with an RTX3090 GPU. The output looks like the following.
{{< video src="yolox_gpu.mp4" width="700px" loop="true" autoplay="true">}}

Out of the box, the model ran at about 40+ FPS on a RTX3090 GPU.
But, on a Core i9-11900 CPU (a relatively powerful CPU to date) it maxed out at around 7+ FPS - not ideal for a real-time detection.

{{< video src="yolox_cpu.mp4" width="700px" loop="true" autoplay="true">}}

Now, let's improve that by optimizing the model.

### 🤖 ONNX Runtime
{{< figure_resizing src="onnx_runtime.png">}}
ONNX is an open format used to represent machine learning models.
The goal of ONNX is to ensure interoperability among machine learning models via commonly accepted standards.
This allows developers to flexibly move between frameworks such as PyTorch or Tensorflow with less to worry about compatibility.

ONNX also supports cross-platform model accelerator known as ONNX Runtime.
This improves the inference performance of a wide variety of models capable of running on various operating systems.

We can now convert our trained `YOLOX-s` model into ONNX format and run it using the ONNX Runtime.
Before that you must install the `onnxruntime` package via `pip`.

```bash
pip install onnxruntime
```

To convert our model run

```bash
python tools/export_onnx.py --output-name your_yolox.onnx -f exps/your_dir/your_yolox.py -c your_yolox.pth
```
More details can be found [here](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime).
Let's run the inference now using the ONNX model and ONNX Runtime.

{{< video src="onnx.mp4" width="700px" loop="true" autoplay="true">}}

As you can see, the FPS slightly improved from 7+ FPS to 11+ FPS with the ONNX model and Runtime on CPU - still it's not very ideal for real-time inference.

Let's see if we can improve that further.


### 🔗 OpenVINO Intermediate Representation
{{< figure_resizing src="openvino_logo.png">}}

```bash
pip install openvino-dev[onnx]==2022.1.0
```

```bash
mo --input_model models/ONNX/yolox_s_lp.onnx --input_shape [1,3,640,640] --data_type FP16 --output_dir models/IR/
```

Now, let's run the inference on the same video and observe its performance.
{{< video src="fp16.mp4" width="700px" loop="true" autoplay="true">}}

As you can see the FPS bumped up to 19+ FPS! It's now beginning to look more feasible for a real-time detection.
Let's call it a day and celebrate the success of our model!

Or, is there more to it? Enter 👇👇👇

### 🛠 Post-Training Quantization
Now this is where the real magic happens.

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





