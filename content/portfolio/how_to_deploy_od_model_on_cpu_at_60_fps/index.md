---
title: "How to 10x your Object Detection Model and Deploy on CPU at 50+ FPS"
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

### 🚦 Motivation

{{< notice tip >}}
By the end of this post, you will learn how to:

* Train state-of-the-art YOLOX model with your own data.
* Convert the YOLOX PyTorch model into ONNX and OpenVINO IR format.
* Run quantization algorithm to 10x your model's inference speed. 

**P/S**: The final model runs faster on the CPU than the GPU! 😱
{{< /notice >}}


Deep learning (DL), they seem to be the magic word that makes anything cool. 
We find them everywhere - in blog posts, articles, research papers, advertisements and even [baby books](https://www.amazon.com/Neural-Networks-Babies-Baby-University/dp/1492671207). 

Except in production 🤷‍♂️.

As much as we were made to believe DL is the answer to our problems, more than 85% of models don't make it into production - according to a recent survey by [Gartner](https://www.gartner.com/en/newsroom/press-releases/2018-02-13-gartner-says-nearly-half-of-cios-are-planning-to-deploy-artificial-intelligence).

The barrier? *Deployment*.

In object detection, we typically train models on massive GPUs either locally or in the cloud.
But when it comes to deployment, running them on GPUs is often impractical.

On the other hand, CPUs are far more common in deployment, and a lot cheaper. 
But can we feasibly deploy real-time DL models on a CPU?
Running DL models on a CPU is orders of magnitude slower compared to GPU, right?

**Wrong**.

 
In this post, I will walk you through how we go from this 🐌🐌🐌

{{< video src="yolox_cpu.mp4" width="700px" loop="true" autoplay="true">}}

to this 🚀🚀🚀
{{< video src="int8.mp4" width="700px" loop="true" autoplay="true">}}

Yes, you saw that right, this model runs on a CPU 😱.
If that looks interesting, let's dive in 👇.


### ⛷ Modeling with YOLOX
{{< figure_resizing src="yolox_demo.png">}}

We will use a state-of-the-art [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) model to detect the license plate of vehicles around the neighborhood.
YOLOX is one of the most recent YOLO series model that is both lightweight and accurate.

It claims better performance than [YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4), [YOLOv5](https://github.com/ultralytics/yolov5), and [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) models.
Additionally, YOLOX is a anchorless one-stage detector which makes it faster that its counterparts.

Before we start training, let's collect images of the license plate and annotate them.
I collected about 40 images in total. 
30 of the images will be used as the training set and 10 as the validation set.
These are incredibly small sample size for any DL model, but I found that it works reasonably well for our task at hand.

{{< figure_resizing src="sample_imgs.png" caption="Sample images of vehicle license plates.">}}


To label the images, let's use the open-source [CVAT](https://github.com/openvinotoolkit/cvat) labeling tool by Intel.
There are a ton of other labeling tools out there feel free to use them if you are comfortable.

If you'd like to try CVAT, head to https://cvat.org/ - create an account and log in.
No installation needed.

A top menu bar should be visible as shown below. 
Click on *Task*, fill in the name of the task, add related labels and upload the images.
{{< figure_resizing src="cvat_new.png">}}

Since we are only interested in labeling the license plate, I've entered only one label - `LP` (license plate).
Once the uploading completes, you will see a summary page as below. 
Click on `Job #368378` at ③ and it should bring you to the labeling page.
{{< figure_resizing src="task_description.png">}}

To start labeling, click on the square icon at ① and click Shape at ② in the figure below.
{{< figure_resizing src="draw_box.png">}}

You can then start drawing bounding boxes around the license plate. Do this for all 40 images.
{{< figure_resizing src="show_bbox.png">}}

Once done, we are ready to export the annotations on the *Tasks* page.
{{< figure_resizing src="export.png">}}

Make sure the format is *COCO 1.0* and click on OK. If you'd like to download the images check the Save images box. Since I have those images already, I don't have to download them.
{{< figure_resizing src="coco.png">}}


Now, that we have our dataset ready, let's begin training.
If you haven't already, install the YOLOX package by following the instructions [here](https://github.com/Megvii-BaseDetection/YOLOX).

Place your images and annotations in the `datasets` folder following the structure outlined [here](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/datasets).

Next, we must also prepare a custom `Exp` class.
The `Exp` class is a `.py` file where we can configure everything about the model - dataset location, data augmentation, model architecture, and other training hyperparameters.
More info on the `Exp` class [here](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/docs/train_custom_data.md). 


For simplicity let's use one of the smaller models - `YOLOX-s`.

There are a host of other YOLOX models you can try like `YOLOX-m`, `YOLOX-l`, `YOLOX-x`, `YOLOX-Nano`, `YOLOX-Tiny`, etc. 
Feel free to experiment with them.
More details are on the [README](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/README.md) of the YOLOX repo. 

My custom `Exp` class for the `YOLOX-s` model looks like the following

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

We are now ready to start the training. 
The training script is provided in the `tools` folder. To start training, run 

```bash
python tools/train.py -f exps/example/custom/yolox_s.py -d 1 -b 64 --fp16 -o -c /path/to/yolox_s.pth
```

{{< notice note >}}
+ `-f` specifies the location of the custom `Exp` file.

+ `-d` specifies the number of GPUs available on your machine.

+ `-b` specifies the batch size.

+ `-c` specifies the path to save your checkpoint.

+ `--fp16` tells the model to train in mixed precision mode.

{{< /notice >}}


After the training completes, you can use run an inference on the model by utilizing the `demo.py` script in the same folder.

```bash
python tools/demo.py video -f exps/example/custom/yolox_s.py -c /path/to/your/yolox_s.pth --path /path/to/your/video --conf 0.25 --nms 0.45 --tsize 640 --device [cpu/gpu]
```

{{< notice note >}}
+ `-f` specifies the path to the custom `Exp` file.

+ `--path` specifies the path to the saved checkpoint file.

+ `--conf` specifies the confidence threshold of the detection.

+ `--nms` specifies the non-maximum suppression threshold.

+ `--tsize` specifies the test image size.

+ `--device` specifies the device to run the model on - `cpu` or `gpu`.

{{< /notice >}}

I'm running this on a computer with an RTX3090 GPU. The output looks like the following.
{{< video src="yolox_gpu.mp4" width="700px" loop="true" autoplay="false">}}

Out of the box, the model averaged at 40+ FPS on a RTX3090 GPU.
But, on a Core i9-11900 CPU (a relatively powerful CPU to date) it averaged at 5+ FPS - not good for a real-time detection task.

{{< video src="yolox_cpu.mp4" width="700px" loop="true" autoplay="false">}}

Now, let's improve that by optimizing the model.

### 🤖 ONNX Runtime
{{< figure_resizing src="onnx_runtime.png">}}
[ONNX](https://onnx.ai/) is an open format built to represent machine learning models.
The goal of ONNX is to ensure interoperability among machine learning models via commonly accepted standards.
This allows developers to flexibly move between frameworks such as PyTorch or Tensorflow with less to worry about compatibility.

ONNX also supports cross-platform model accelerator known as ONNX Runtime.
This improves the inference performance of a wide variety of models capable of running on various operating systems.

We can now convert our trained `YOLOX-s` model into ONNX format and run it using the ONNX Runtime.
For that, you must install the `onnxruntime` package via `pip`. 

```bash
pip install onnxruntime
```

To convert our model run

```bash
python tools/export_onnx.py --output-name your_yolox.onnx -f exps/your_dir/your_yolox.py -c your_yolox.pth
```

Let's run the inference now using the ONNX model and ONNX Runtime.

{{< video src="onnx.mp4" width="700px" loop="true" autoplay="false">}}

As you can see, the FPS slightly improved from 5+ FPS to about 10+ FPS with the ONNX model and Runtime on CPU - it's still not ideal for real-time inference.
Just by converting the model to ONNX, we already 2x the inference performance. 

Let's see if we can improve that further.


### 🔗 OpenVINO Intermediate Representation
{{< figure_resizing src="openvino_logo.png">}}

OpenVINO is a toolkit to facilitate optimization of DL models to be run on Intel hardware.
The toolkit enables a model to be optimized once and deployed on any supported Intel hardware including CPU, GPU, VPU and FPGAs.

To optimize performance, the model needs to be converted into a form known as the OpenVINO Intermediate Representation (IR). 
This consist of a `.xml` and `.bin` file.
With the IR files, you can deploy them on any of the supported Intel hardware.

Let's now convert our model into the IR form. 
For that, we need to install the `openvino-dev` package.

```bash
pip install openvino-dev[onnx]==2022.1.0
```
Once installed, we can invoke the `mo` command to convert the model. `mo` is the abbreviation of Model Optimizer. 

```bash
mo --input_model models/ONNX/yolox_s_lp.onnx --input_shape [1,3,640,640] --data_type FP16 --output_dir models/IR/
```
{{< notice note >}}
The `mo` accepts a few parameters:

+ `--input_model` specifies the path to the previously converted ONNX model.

+ `--input_shape` specifies the shape of the input image.

+ `--data_type` specifies the data type.

+ `--output_dir` specifies the directory to save the IR.
{{< /notice >}}

Now, let's run the inference on the same video and observe its performance.
{{< video src="fp16.mp4" width="700px" loop="true" autoplay="false">}}

As you can see the FPS bumped up to 16+ FPS. 
It's now beginning to look more feasible for a real-time detection.
Let's call it a day and celebrate the success of our model! 🥳

Or, is there more to it? Enter 👇👇👇

### 🛠 Post-Training Quantization
Apart from the Model Optimizer, OpenVINO also comes with a Post-training Optimization Toolkit (POT) designed to supercharge the inference of DL models without retraining or finetuning.

To achieve that, POT runs 8-bit quantization algorithms and optimizes the model to use integer tensor instead of floating point tensor on some operations.
This results in a 2-4x faster and smaller model.

Now this is where the real magic happens.

From the OpenVINO documentation [page](https://docs.openvino.ai/2021.1/pot_compression_algorithms_quantization_README.html), the POT supports two types of quantization:

+ `DefaultQuantization` is a default method that provides fast and in most cases accurate results for 8-bit quantization. 

+ `AccuracyAwareQuantization` enables remaining at a predefined range of accuracy drop after quantization at the cost of performance improvement. It may require more time for quantization.

The quantization algorithm requires a representative subset of the dataset to estimate the model accuracy during the quantization process.
Let's prepare a few images and put them in a separate directory.

We are going to use the `DefaultQuantization` in this post.
For that, let's run 


```bash
pot -q default -m models/IR/yolox_s_lp.xml -w models/IR/yolox_s_lp.bin --engine simplified --data-source data/pot_images --output-dir models/INT8
```
{{< notice note >}}
The `pot` command accepts a few parameters:

+ `-m` specifies the directory to the `.xml` file.
+ `-w` specifies the directory to the `.bin` file.
+ `--engine` specifies the type of quantization algorithm.
+ `--data-source` specifies the directory of the images used to quantize the model.
+ `--output-dir` specifies the directory to save the quantized model.
{{< /notice >}}

This results in another set of IR files saved in `--output-dir`. 

### 🚀 Real-time Inference @50+ FPS
Now, the moment of truth.
Let's load the quantized model and run the same inference again.

{{< video src="int8.mp4" width="700px" loop="true" autoplay="false" >}}

The first time I saw the numbers, I could hardly believe my eyes. 50+ FPS on a CPU! 
That's about 10x faster compared to our initial model! 🚀
Additionally, this is also faster than the model ran on the RTX3090 GPU!

### 🏁 Conclusion
{{< notice tip >}}
In this post you've learned how to: 

+ Train a custom YOLOX model with your own dataset.
+ Convert the trained model into ONNX and IR forms for inference.
+ 10x the inference speed of the model with 8-bit quantization.
{{< /notice >}}

What's next? You can experiment with using other YOLOX models and see if you can squeeze more using the techniques covered in this post.
You can also try the `AccuracyAwareQuantization` which runs quantization on the model with lesser accuracy loss on the model.
There also a best practice guide to quantizing your model with OpenVINO [here](https://docs.openvino.ai/latest/pot_docs_BestPractices.html).

### 🙏 Comments & Feedback
If you like this and don't want to miss any of my future content, follow me on [Twitter](https://twitter.com/dicksonneoh7) and [LinkedIn](https://www.linkedin.com/in/dickson-neoh/) where I share more of these in bite-size posts.

I hope you've learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter post or [drop me a message](https://dicksonneoh.com/contact/).
{{< tweet 1517004585495240704>}}





