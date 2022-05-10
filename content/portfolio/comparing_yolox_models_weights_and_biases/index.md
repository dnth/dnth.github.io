---
title: "Choosing the Best YOLOX Model with Weights and Biases."
date: 2022-01-07T15:00:15+08:00
featureImage: images/portfolio/comparing_yolox_models_with_weights_and_biases/thumbnail.gif
postImage: images/portfolio/comparing_yolox_models_with_weights_and_biases/post_image.png
tags: ["OpenVINO", "YOLOX", "Wandb", "real-time", "optimization", "license-plate"]
categories: ["deployment", "object-detection", "monitoring"]
toc: true
socialshare: true
description: "Monitor your models with Wandb and get your life back!"
images : 
- images/portfolio/comparing_yolox_models_with_weights_and_biases/post_image.png
---
{{< notice info >}}
This blog post is still a work in progress. If you require further clarifications before the contents are finalized, please get in touch with me [here](https://dicksonneoh.com/contact/), on [LinkedIn](https://www.linkedin.com/in/dickson-neoh/), or [Twitter](https://twitter.com/dicksonneoh7).
{{< /notice >}}

### 🔎 Motivation

{{< notice tip >}}
By the end of this post you will learn how to:
+ Install the Wandb client and log the YOLOX training metrics to Wandb.
+ Compare training metrics on Wandb dashboard.
+ Picking the best model from mAP and FPS values.
{{< /notice >}}

"**So many models, so little time!**"

As a machine learning engineer, I often hear this phrase thrown around in many variations.

In object detection alone, there are already several hundreds of models out there. 
With each passing day, better models are added as new discoveries are made.
If you're new, this can easily get overwhelming.

Even within the YOLOX series there are 6 variations of the model to choose from.
There are 3 questions that beg answering:

+ How do you pick the best model?
+ How do the models compare to one another?
+ How do you keep track and log the performance of each model?

In this blog post I will show you how I accomplish all of them by using a free and simple tool from [Weights and Biases](https://wandb.ai/home) (Wandb) 👇

**PS**: No Excel sheets involved.

### 📉 Wandb - Google Drive for Machine Learning


Life is short they say. So why waste it on monitoring your deep learning models when you can automate them?
This is what Wandb is trying to solve. It's like Google Drive for machine learning.

Wandb helps individuals and teams build models faster.
With just few lines of code, you can compare models, log important metrics, and collaborate with teammates.
It's free to get started. Click [here](https://wandb.ai/) to create an account. 

{{< figure_resizing src="wandb.png">}}

This post is a sequel to my previous post where I showed [how to deploy YOLOX models on CPU at 50 FPS](https://dicksonneoh.com/portfolio/how_to_10x_your_od_model_and_deploy_50fps_cpu/).
This time around I will show you how I get the most from the YOLOX models by logging the performance metrics and comparing them on Wandb.

Let's first install the Wandb client for `Python`:

``` bash
pip install wandb
```

Next, run 
```bash
wandb login
```
from your terminal to authenticate your machine. The API key is stored in `~/.netrc`.

### 👀 Monitoring Training Metrics
Once installed you can use Wandb to monitor the training metrics of YOLOX.
All you need to do is run the `train.py` script from the YOLOX repository and specify `wandb` as the argument.

```bash
python tools/train.py -f exps/example/custom/yolox_s.py -d 1 -b 64 --fp16 -o -c /path/to/yolox_s.pth --logger wandb wandb-project yolox-compare-blog wandb-id yolox-x-640
```

{{< notice note >}}
+ `-f` specifies the location of the custom `Exp` file.

+ `-d` specifies the number of GPUs available on your machine.

+ `-b` specifies the batch size.

+ `-c` specifies the path to save your checkpoint.

+ `--fp16` tells the model to train in mixed precision mode.

+ `--logger` specifies the type of logger we want to use. The default is `tensorboard`.

+ `wandb-project` specifies the name of the project on Wandb.

+ `wandb-id` specifies the id of the run.

{{< /notice >}}

I also recommend you specify `self.save_history_ckpt = False` in your `Exp` file.
If set to `True` this saves the model checkpoint at every epoch and uploads them to `wandb`.
This makes the logging process slower because every checkpoint is uploaded to `wandb` as an artifact.


Once everything is set in place, let's run the training script and head to the project dashboard on Wandb to monitor the logged metrics.
The project dashboard should look like the following. You can access the dashboard for this project [here](https://wandb.ai/dnth/yolox-compare-blog?workspace=user-dnth).

{{< figure_resizing src="graphs.png" >}}

You can run the `train.py` on multiple YOLOX models and the metrics should show up on the same dashboard.
I ran the training on my license plate dataset for all YOLOX models namely `YOLOX-Nano`, `YOLOX-Tiny`, `YOLOX-S`, `YOLOX-M`, `YOLOX-L` and `YOLOX-X`.


To gauge the quality of the model, we can look at the `COCOAP50_95` plot or also known as the COCO Metric or mean average precision (mAP) plot.
This plot shows how well the model performs on the validation set as we train the model.
{{< figure_resizing src="mAP.png" >}}

Looks like `YOLOX-X` got the highest score followed by `YOLOX-L`, `YOLOX-M`, `YOLOX-S`, `YOLOX-Tiny` and `YOLOX-Nano`.
It's a little hard to tell from the figure above.
I encourage you to checkout the [dashboard](https://wandb.ai/dnth/yolox-compare-blog?workspace=user-dnth) where you can zoom in and resize the plots.




### ⚖️ Comparing YOLOX Models
Most of the time it is not enough to just compare the models solely on the mAP values.
In object detection, it is always good to verify the performance by visual inspection of the model.
For that, let's run an inference on a video for each model.

Running inference on a GPU is boring, we know YOLOX models can run very fast on GPUs.
To make it more interesting, let's run the models on a CPU instead.
Before that, let's convert the YOLOX models into a form that can run efficiently on CPUs.

For that, we use Intel's Post-training Optimization Toolkit (POT) that runs an `INT8` quantization algorithm on the YOLOX models.
Quantization optimizes the model to use integer tensors instead of floating-point tensors.
This results in a **2-4x faster and smaller models**.
Plus, we can now run the models on a CPU!

If you're new to my posts, I wrote on how to run the quantization [here](https://dicksonneoh.com/portfolio/how_to_10x_your_od_model_and_deploy_50fps_cpu/).

Let's checkout how the models perform running on a CPU 👇

#### YOLOX-X (mAP: 0.8869)
{{< video src="vids/yolox_x.mp4" width="700px" loop="true" autoplay="false" >}}
#### YOLOX-L (mAP: 0.8729)
{{< video src="vids/yolox_l.mp4" width="700px" loop="true" autoplay="false" >}}
#### YOLOX-M (mAP: 0.8688)
{{< video src="vids/yolox_m.mp4" width="700px" loop="true" autoplay="false" >}}
#### YOLOX-S (mAP: 0.8560)
{{< video src="vids/yolox_s.mp4" width="700px" loop="true" autoplay="false" >}}
#### YOLOX-Tiny (mAP: 0.8422)
{{< video src="vids/yolox_tiny.mp4" width="700px" loop="true" autoplay="false" >}}
#### YOLOX-Nano (mAP: 0.7905)
{{< video src="vids/yolox_nano.mp4" width="700px" loop="true" autoplay="false" >}}

Observing carefully, we notice that as the models get smaller, the mAP decreases and the FPS increases.
Which model to use will ultimately depend on your application.

Need something really fast on the edge with a little compromise in accuracy, get YOLOX-Nano.
If you don't need real-time inference on the edge and require an accurate model then YOLOX-X fits the description. 

This is a classic trade-off of accuracy vs latency in machine learning. Knowing your application well can help you pick the best model for the job.

### ⛳️ Wrapping up
That's it! In this blog post I have shown you how easy it is to use Wandb to log the training metrics of your YOLOX models.
Also we compared all the quantized YOLOX models and it's performance on a CPU.

{{< notice tip>}}
In this post I've shown you how to

+ Install wandb client.
+ Use Wandb to log and compare training metrics.
+ Picking the best model using mAP values and visual inspection.

{{< /notice >}}

So, what's next? In this short post, we have not explored all features available on Wandb.
As next steps, I encourage you to check the Wandb [documentation](https://docs.wandb.ai/) page to see what's possible.

Here are my 3 suggestions:

+ Try out hyperparameter optimization with Sweeps.
+ Learn how to version your model and data.
+ Learn how to visualize and inspect your data.


### 🙏 Comments & Feedback
I hope you've learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter post or [drop me a message](https://dicksonneoh.com/contact/).
{{< tweet dicksonneoh7 1521342853640728576>}}


If you like what you see and don't want to miss any of my future contents, follow me on Twitter and LinkedIn where I deliver more of these tips in bite-size posts.
