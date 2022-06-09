---
title: "Supercharging YOLOv5: How I Got 182.4 FPS Inference Without a GPU"
date: 2022-06-07T11:00:15+08:00
featureImage: images/portfolio/supercharging_yolov5/thumbnail.gif
postImage: images/portfolio/supercharging_yolov5/post_image.png
tags: ["DeepSparse", "ONNX", "YOLOv5", "real-time", "optimization", "pistol"]
categories: ["deployment", "object-detection", "modeling"]
toc: true
socialshare: true
description: "Accelerate inference up to 180+ FPS on a CPU! For free!"
images : 
- images/portfolio/supercharging_yolov5/post_image.png
---


### 🔥 Motivation
After months of searching, you've finally found *the one*. 

The one object detection library that just works.
No installation hassle, no package version mismatch, and no `CUDA` errors. 

I'm talking about the amazingly engineered [YOLOv5](https://github.com/ultralytics/yolov5) object detection library by [Ultralytics](https://ultralytics.com/yolov5).

Elated, you quickly find an interesting dataset from [Roboflow](https://roboflow.com/) and finally trained a state-of-the-art (SOTA) YOLOv5 model to detect firearms from image streams.

You ran through a quick checklist --
+ Inference results, checked ✅
+ `COCO` mAP, checked ✅
+ Live inference latency, checked ✅

You're on top of the world. 

<iframe src="https://giphy.com/embed/zEJRrMkDvRe5G" width="480" height="360" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/win-zEJRrMkDvRe5G"></a></p>


You can finally pitch the results to your clients next Monday.
At the back of your mind, you can already see your clients' impressed look on the astonishing feat.

On the pitching day, just when you thought things are going in the right direction.
One of the clients asked,

"**Does your model run on our existing CPU?**"

You flinched. 

That wasn't something you anticipated. You tried to convince them that GPUs are *"the way forward"* and it's *"the best way"* to run your model in real-time.

You scanned the room and begin to notice the stiff looks on their faces 👇

{{< figure_resizing src="meme.jpg">}}

Needless to say it didn't go well.
I hope nobody will have to face this awkward situation in a pitching session, ever.
You don't have to learn it the hard way, like I did.

You may wonder, can we really use consumer grade CPUs to run models in real-time?

🦾**YES we can!**

I wasn't a believer, but now I am, after discovering [Neural Magic](https://neuralmagic.com/).

In this post, I show you how you can supercharge your YOLOv5 inference performance running on CPUs using **free** and open-source tools by Neural Magic.

{{< notice tip >}}
By the end of this post, you will learn how to:

* Train a SOTA YOLOv5 model on your own data.
* Sparsify the model using SparseML quantization aware training, sparse transfer learning, and one-shot quantization.
* Export the sparsified model and run it using the DeepSparse engine at insane speeds. 

**P/S**: The end result - YOLOv5 on CPU at 180+ FPS using only 4 CPU cores! 🚀
{{< /notice >}}

If that sounds exciting let's dive in 🧙


### 🔩 Setting Up


#### 🔫 Dataset

The [recent gun violence](https://edition.cnn.com/2022/05/25/us/uvalde-texas-elementary-school-shooting-what-we-know/index.html) news had me thinking deeply about how we can prevent incidents like these again. 
This is the worst gun violence since 2012, and 21 innocent lives were lost.

I'm deeply saddened, and my heart goes out to all victims of the violence and their loved ones.

I'm not a lawmaker, so there is little I can do there. 
But, I think I know something in computer vision that might help.
That's when I came across the [Pistols Dataset](https://public.roboflow.com/object-detection/pistols) from Roboflow.


This dataset contains 2986 images and 3448 labels across a single annotation class: pistols. 
Images are wide-ranging: pistols in hand, cartoons, and staged studio-quality images of guns. 
The dataset was originally released by the University of Grenada.

{{< figure_resizing src="pistol.png">}}

#### 🦸 Installation 

Now let's put the downloaded Pistols Dataset into the appropriate folder first.
I will put the downloaded images and labels into the `datasets/` folder.

Let's also put the sparsification recipes from [SparseML](https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov5/recipes) into the `recipes/` folder. More on `recipes` later.

Here's a high-level overview of my directory.

```tree
├── req.txt
├── datasets
│   ├── pistols
│   │   ├── train
|   |   ├── valid
├── recipes
│   ├── yolov5s.pruned.md
│   ├── yolov5.transfer_learn_pruned.md
│   ├── yolov5.transfer_learn_pruned_quantized.md
|   └── ...
└── yolov5-train
        ├── data
        |   ├── hyps
        |   |   ├── hyps.scratch.yaml
        |   |   └── ...
        |   ├── pistols.yaml
        |   └── ...
        ├── models_v5.0
        |   ├── yolov5s.yaml
        |   └── ...
        ├── train.py
        ├── export.py
        ├── annotate.py
        └── ...
```

{{< notice note >}}

+ `req.txt` - Requirement file to install all packages used in this post.

+ `datasets/` - Contains the train and validation images/labels downloaded from Roboflow.

+ `recipes/` - Contains sparsification recipes from the [SparseML](https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov5/recipes) repo.

+ `yolov5-train/` - Cloned directory from Neural Magic's YOLOv5 [fork](https://github.com/neuralmagic/yolov5). 

**NOTE**: You can explore further into the folder structure on my [Github repo](https://github.com/dnth/yolov5-deepsparse-blogpost).
{{< /notice >}}

{{< notice warning >}}
**IMPORTANT**: The sparsification recipes will only work with Neural Magic's YOLOv5 fork and will **NOT WORK** with the original YOLOv5 by Ultralytics.
{{< /notice >}}

For this post, we are going to use a [forked version](https://github.com/neuralmagic/yolov5) of the YOLOv5 library that will allow us to do custom optimizations in the upcoming section.

To install, run the following commands

```bash
git clone https://github.com/neuralmagic/yolov5.git
cd yolov5
git checkout release/0.12
pip install -r requirements.txt
```

To install all remaining packages used in this post, run
```bash
pip install -r req.txt
```





### ⛳ Baseline Performance

#### 🔦 PyTorch

Now that everything's in place, let's start by training a baseline model with no optimization.

For that, run the `train.py` script in the `yolov5-train/` folder.
```bash
python train.py --cfg ./models_v5.0/yolov5s.yaml \
                --data pistols.yaml \
                --hyp data/hyps/hyp.scratch.yaml \
                --weights yolov5s.pt --img 416 --batch-size 64 \
                --optimizer SGD --epochs 240 \
                --project yolov5-deepsparse --name yolov5s-sgd
```

{{< notice note >}}
+ `--cfg` -- Path to the configuration file which stores the model architecture.

+ `--data` -- Path to the `.yaml` file that stores the details of the Pistols dataset.

+ `--hyp` -- Path to the `.yaml` file that stores the training hyperparameter configurations.

+ `--weights` -- Path to a pretrained weight.

+ `--img` -- Input image size.

+ `--batch-size` -- Batch size used in training.

+ `--optimizer` -- Type of optimizer. Options include `SGD`, `Adam`, `AdamW`.

+ `--epochs` -- Number of training epochs.

+ `--project` -- Wandb project name.

+ `--name` -- Wandb run id.

{{< /notice >}}

All metrics are logged to Weights & Biases (Wandb) [here](https://wandb.ai/dnth/yolov5-deepsparse).

Once training's done, let's run inference on a video with the `annotate.py` script.

```bash
python annotate.py yolov5-deepsparse/yolov5s-sgd/weights/best.pt \
                --source data/pexels-cottonbro-8717592.mp4 \
                --engine torch \
                --image-shape 416 416 \
                --device cpu \
                --conf-thres 0.7
```

{{< notice note >}}
The first argument points to the `.pt` saved checkpoint.

+ `--source` - The input to run inference on. Options: path to video/images or just specify `0` to infer on your webcam.

+ `--engine` - Which engine to use. Options: `torch`, `deepsparse`, `onnxruntime`.

+ `--image-size` -- Input resolution.

+ `--device` -- Device to use for inference. Options: `cpu` or `0` (GPU).

+ `--conf-thres` -- Confidence threshold for inference.

**NOTE**: The inference output will be saved in the `annotation_results/` folder.

{{< /notice >}}

Here's how it looks like running the baseline YOLOv5-S on an Intel i9-11900 using all 8 CPU cores.

{{< video src="vids/torch-annotation/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

+ Average FPS : 21.91
+ Average inference time (ms) : 45.58

<!-- And for reference, inference on an RTX3090 GPU.

{{< video src="vids/torch-gpu/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

+ Average FPS : 89.20
+ Average inference time (ms) : 11.21 -->

Actually, the FPS looks quite decent already and might suit some applications even without further optimization.

But why settle when you can get something better? 
After all, that's why you're here, right? 😉

Meet 👇

#### 🕸 DeepSparse Engine
DeepSparse is an inference engine by Neural Magic that runs optimally on CPUs.
It's incredibly easy to use. Just give it an ONNX model and you're ready to roll.

Let's export our `.pt` file into ONNX using the `export.py` script.

```bash
python export.py --weights yolov5-deepsparse/yolov5s-sgd/weights/best.pt \
                --include onnx \
                --imgsz 416 \
                --dynamic \
                --simplify
```

{{< notice note >}}
`--weight` -- Path to the `.pt` checkpoint.

`--include` -- Which format to export to. Options: `torchscript`, `onnx`, [etc](https://github.com/dnth/yolov5-deepsparse-blogpost/blob/4d44b32909bbc9e8b3bb7f8bf89f0e50361872f7/yolov5-train/export.py#L694).

`--imgsz` -- Image size.

`--dynamic` -- Dynamic axes.

`--simplify` -- Simplify the ONNX model.

{{< /notice >}}

And now, run the inference script again, this time using the `deepsparse` engine and with only 4 CPU cores in the `--num-cores` argument.

```bash
python annotate.py yolov5-deepsparse/yolov5s-sgd/weights/best.onnx \
        --source data/pexels-cottonbro-8717592.mp4 \
        --image-shape 416 416 \
        --conf-thres 0.7 \
        --engine deepsparse \
        --device cpu \
        --num-cores 4
```

{{< video src="vids/onnx-annotation/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

+ Average FPS : 29.48
+ Average inference time (ms) : 33.91

Just like that, we improved the average FPS from 21+ (PyTorch engine on CPU using 8 cores) to 29+ FPS. 
All we did was use the ONNX model with the DeepSparse engine.

**P/S**: We are done with **just the baselines** here! 
The real action only happens next - when we run sparsification with 👇

### 👨‍🍳 SparseML and Recipes

<a href="https://docs.neuralmagic.com/sparseml/"> <img alt="SparseML Flow" src="https://docs.neuralmagic.com/docs/source/infographics/sparseml.png" width="700"/> </a>







Sparsification is the process of removing redundant information from a model.
The result is a **smaller and faster** model. 

This is how we speed up our YOLOv5 model, by a lot!

{{< notice tip>}}
In general, there are 2 methods to sparsify a model - Pruning and Quantization.

+ ✂ Pruning - Removing unused weights in the model.

+ 🔮 Quantization - Forcing a model to use a less accurate storage format i.e. from 32-bit floating point (FP32) to 8-bit integer (INT8).

Used together or separately, this results in a smaller and faster model.

{{< /notice >}}


How do we do it? It's by using [SparseML](https://github.com/neuralmagic/sparseml) - an open-source library by Neural Magic.
With SparseML you can sparsify neural networks by applying pre-made **recipes** to the model. 
You can also modify the recipes to suit your needs.




<!-- It currently supports integration with several well known libraries from computer vision and natural language processing domain. -->






{{< notice note >}}

There are 3 methods to sparsify models with SparseML:

1️⃣ Post-training (One-shot).

2️⃣ Sparse Transfer Learning.

3️⃣ Training Aware.


**NOTE**: 1️⃣ does not require re-training but only supports dynamic quantization. 
2️⃣ and 3️⃣ requires re-training and supports pruning and quantization which may give better results.

{{< /notice >}}

You may wonder, this sounds too good to be true! 

*What's the caveat*?

Good question!

With sparsification, you can expect a slight loss in accuracy depending on the degree of sparsification.
Highly sparse models are usually less accurate than the original model but gains significant boost in speed and latency.

With the recipes from SparseML, the loss of accuracy ranges from 2% to 6%.
In other words the *recovery* is 94% to 98% compared to the performance of the original model.
In exchange, we gain phenomenal speedups, ranging from 2x to 10x faster!

In most situations, this is not a big deal. 
If the accuracy loss is something you can tolerate, then let's sparsify some models already! 🤏.

#### ☝️ One-Shot
The one-shot method is the easiest way to sparsify an existing model as it doesn't require re-training.

But this only works well for dynamic quantization, for now.
There are ongoing works in making one-shot work well for pruning.

Let's run the one-shot method on the baseline model we trained earlier.
All you need to do is add a `--one-shot` argument to the training script, and specify a pruning `--recipe`.
Remember to specify `--weights` to the location of the best checkpoint from the training.

```bash
python train.py --cfg ./models_v5.0/yolov5s.yaml \
                --recipe ../recipes/yolov5s.pruned.md \
                --data pistols.yaml --hyp data/hyps/hyp.scratch.yaml \
                --weights yolov5-deepsparse/yolov5s-sgd/weights/best.pt \
                --img 416 --batch-size 64 --optimizer SGD --epochs 240 \
                --project yolov5-deepsparse --name yolov5s-sgd-one-shot \
                --one-shot
```

It should generate another `.pt` in the directory specified in `--name`.
This `.pt` file stores the quantized weights in `int8` format instead of `fp32` resulting in a reduction in model size and inference speedups.

Next, let's export the quantized `.pt` file into `ONNX` format.

```bash
python export.py --weights yolov5-deepsparse/yolov5s-sgd-one-shot/weights/checkpoint-one-shot.pt \
                 --include onnx \
                 --imgsz 416 \
                 --dynamic \
                 --simplify
```

And run an inference 

```bash
python annotate.py yolov5-deepsparse/yolov5s-sgd-one-shot/weights/checkpoint-one-shot.onnx \
                --source data/pexels-cottonbro-8717592.mp4 \
                --image-shape 416 416 \
                --conf-thres 0.7 \
                --engine deepsparse \
                --device cpu \
                --num-cores 4
```



{{< video src="vids/one-shot/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

+ Average FPS : 32.00
+ Average inference time (ms) : 31.24

At no re-training cost, we are performing 10 FPS better than the original model.
We maxed out at about 40 FPS!

The one-shot method only took seconds to complete.
If you're looking for the easiest method for performance gain, one-shot is the way to go.

But, if you're willing to re-train the model to double its performance and speed, read on 👇

#### 🤹‍♂️ Sparse Transfer Learning
With SparseML you can take an already sparsified model (pruned and quantized) and fine-tune it on your own dataset.
This is known as *Sparse Transfer Learning*.

This can be done by running

```bash
python train.py --data pistols.yaml --cfg ./models_v5.0/yolov5s.yaml 
                --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94?recipe_type=transfer 
                --img 416 --batch-size 64 --hyp data/hyps/hyp.scratch.yaml 
                --recipe ../recipes/yolov5.transfer_learn_pruned_quantized.md 
                --optimizer SGD
                --project yolov5-deepsparse --name yolov5s-sgd-pruned-quantized-transfer
```

The above command loads a sparse YOLOv5-S from Neural Magic's [SparseZoo](https://github.com/neuralmagic/sparsezoo) and runs the training on your dataset.

The `--weights` argument points to a model from the SparseZoo.
There are more sparsified [models available](https://docs.neuralmagic.com/sparsezoo/source/models.html) in SparseZoo. 
I will leave it to you to explore which model works best.

Running inference with `annotate.py` results in
{{< video src="vids/yolov5s-pruned-quant-tl/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

+ Average FPS : 51.56
+ Average inference time (ms) : 19.39

We almost 2x the FPS from the previous one-shot method!
Judging from the FPS value and `mAP` [scores]((https://wandb.ai/dnth/yolov5-deepsparse)), Sparse Transfer Learning makes a lot of sense for most applications.

But, if you scrutinize further into the `mAP` metric on the [Wandb dashboard](https://wandb.ai/dnth/yolov5-deepsparse), you'll notice it's slightly lower than the next method 💪.

#### ✂ Pruned YOLOv5-S
Here, instead of taking an already sparsified model, we are going to sparsify our model by pruning it ourselves.

To do that we will use a pre-made recipe on the SparseML [repo](https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov5/recipes).
This recipe tells the training script how to prune the model during training.

For that, we slightly modify the arguments of `train.py`

```bash
python train.py --cfg ./models_v5.0/yolov5s.yaml \
                --recipe ../recipes/yolov5s.pruned.md
                --data pistols.yaml \
                --hyp data/hyps/hyp.scratch.yaml \
                --weights yolov5s.pt --img 416 
                --batch-size 64 --optimizer SGD \
                --project yolov5-deepsparse --name yolov5s-sgd-pruned
```

The only change here is the `--recipe` and the `--name` argument.
Also, there is no need to specify the `--epoch` argument because the number of training epochs is specified in the recipe.


`--recipe` tells the training script which recipe to use for the YOLOv5-S model.
In this case, we are using the `yolov5s.pruned.md` recipe which only prunes the model as it trains.
You can change how aggressive your model is pruned by modifying the `yolov5s.pruned.md` recipe.

{{< notice warning >}}
**IMPORTANT**: The sparsification recipes are model dependent. Eg. YOLOv5-S recipes will not work with YOLOv5-L. 

So make sure you get the right recipe for the right model. Check out other YOLOv5 pre-made recipes [here](https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov5/recipes).
{{< /notice >}}


Running inference, we find

{{< video src="vids/yolov5s-pruned/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

+ Average FPS : 35.50
+ Average inference time (ms) : 31.73

The drop in FPS is expected compared to the Sparse Transfer Learning method because this model is only pruned and **not quantized**. 
But we gain higher `mAP` values. 

But, what if we can run both pruning and quantization? And still score high `mAP` values? 

Of course, why not? 🤖

#### 🪚 Pruned + Quantized YOLOv5-S
Now, let's take it to the next level by running both pruning and quantization.
Note the difference `--recipe` I'm using.

```bash
python train.py --cfg ./models_v5.0/yolov5s.yaml \
                --recipe ../recipes/yolov5.transfer_learn_pruned_quantized.md \
                --data pistols.yaml \
                --hyp data/hyps/hyp.scratch.yaml \
                --weights yolov5s.pt --img 416 \
                --batch-size 64 --optimizer SGD \
                --project yolov5-deepsparse --name yolov5s-sgd-pruned-quantized
```

Export with `export.py` and run inference with `annotate.py`. 
We get

{{< video src="vids/yolov5s-pruned-quant/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

+ Average FPS : 58.06
+ Average inference time (ms) : 17.22

On our Wandb [dashboard](https://wandb.ai/dnth/yolov5-deepsparse?workspace=user-dnth) this model scores the highest `mAP` and is also the fastest. 

It's getting the best of both! 🎯

{{< figure_resizing src="mAP.png">}}

I wanted to end the post here. But there is still this nagging thought that I can't ignore.
It's keeping me awake at night. So I had to do this 🤷‍♂️.

**Just how fast can you run YOLOv5 on CPUs? I mean the maximum possible FPS with DeepSparse.**

This led me to 👇.
### 🚀 Supercharging with Smaller Models
In the YOLOv5 series, the YOLOv5-Nano is the smallest model of all. So in theory, this should be the fastest.

So I'm putting my bets on this model. 
Let's apply the same steps again with the YOLOv5-Nano model.

And

..

...

🚀🚀🚀


{{< video src="vids/yolov5n-pruned-quant/results_.mp4" width="700px" loop="true" autoplay="true" muted="true">}}

+ Average FPS : 101.52
+ Average inference time (ms) : 9.84

🤯 This is mindblowing! The max FPS hit the 180+ range.
I never imagine these numbers are possible, especially using only 4 CPU cores.

Seeing this, I can now sleep peacefully at night 😴

### 🚧 Conclusion
What a journey this has been. 

Gone are the days when we need GPUs to run models in real-time. 
With DeepSparse and SparseML, you can get GPU-class performance on commodity CPUs.

{{< notice tip >}}
In this post I've shown you how to:

* Train a SOTA YOLOv5 model with your own data.
* Sparsify the model using SparseML quantization aware training and one-shot quantization.
* Export the sparsified model and run it using the DeepSparse engine at insane speeds. 

**P/S**: The codes are on my GitHub [repo](https://github.com/dnth/yolov5-deepsparse-blogpost). Check it out.
{{< /notice >}}

If you ever get lost in the commands that I used for this post, fear not.
I listed all commands I used to train all models on the [README](https://github.com/dnth/yolov5-deepsparse-blogpost) of the repo.

Also, feel to use the repo with your own dataset and give it a ⭐ if it helps your work.

### 🙏 Comments & Feedback
I hope you've learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter post or [drop me a message](https://dicksonneoh.com/contact/).
{{< tweet dicksonneoh7 1534395572022480896>}}


If you like what you see and don't want to miss any of my future content, follow me on Twitter and LinkedIn where I deliver more of these tips in bite-size posts.