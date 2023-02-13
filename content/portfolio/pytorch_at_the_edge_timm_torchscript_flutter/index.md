---
title: "PyTorch at the Edge: Deploying Over 964 TIMM Models on Android with TorchScript and Flutter"
date: 2023-02-07T11:00:15+08:00
featureImage: images/portfolio/pytorch_at_the_edge_timm_torchscript_flutter/thumbnail.gif
postImage: images/portfolio/pytorch_at_the_edge_timm_torchscript_flutter/post_image.gif
tags: ["TIMM", "TorchScript", "paddy-disease", "Fastai", "Flutter", "Android", "EdgeNeXt"]
categories: ["deployment", "object-classification", "edge"]
toc: true
socialshare: true
description: "Unlock and Deploy Over 900+ SOTA TIMM models on Android with Torchscript!"
images : 
- images/portfolio/pytorch_at_the_edge_timm_torchscript_flutter/post_image.gif
---


### 🔥 Motivation
You finally got into a Kaggle competition. You found a *getting-started notebook* written by a Kaggle Grandmaster and immediately trained a state-of-the-art (SOTA) image classification model.

After some fiddling, you found yourself in the leaderboard topping the charts with **99.9851247\% accuracy** on the test set 😎!

Proud of your achievement you reward yourself to some rest and a good night's sleep. 
And tomorrow it's time to move on to the next dataset (again).

<!-- And then..

{{< figure_resizing src="meme_sleep.jpg" >}} -->

<!-- I hope this doesn't keep you awake at night like it did for me. -->

With various high level libraries like [Keras](https://keras.io/), [Transformer](https://huggingface.co/docs/transformers/index) and [Fastai](https://www.fast.ai/), the barrier to training a SOTA models has never been lower.

On top of that with platforms like [Google Colab](https://colab.research.google.com/) and [Kaggle](https://www.kaggle.com/), pretty much anyone can train a reasonably good model using an old laptop or even a mobile phone (with some patience).

{{% blockquote %}}
The question is no longer "**can we train a SOTA model?**", but "**what happens after that?**"
{{% /blockquote %}}

Unfortunately, after getting the model trained, most people wash their hands off at this point claiming their model works. 
But, what good would SOTA models do if it's just in notebooks and Kaggle leaderboards?

{{% blockquote %}}
Unless the model is deployed and put to use, it's of little benefit to anyone out there.
{{% /blockquote %}}

{{< figure_resizing src="meme.jpg" >}}

But deployment is painful. Running a model on a mobile phone? 

Forget it 🤷‍♂️.

The frustration is real. I remember spending nights exporting models into `ONNX` and it still failed me.

Deploying models on mobile for edge inference used to be complex. 

Not anymore.

In this post I'm going to show you how you can pick from over 900+ SOTA models on [TIMM](https://github.com/rwightman/pytorch-image-models), train them with best practices with [Fastai](https://www.fast.ai/2020/02/13/fastai-A-Layered-API-for-Deep-Learning/), and deploy them on Android using [Flutter](https://flutter.dev/). 

✅ Yes, for free.

{{< notice tip >}}
⚡ By the end of this post you will learn how to:
+ Load a SOTA classification model from TIMM and train it with Fastai.
+ Export the trained model into TorchScript for inference.
+ Create a functional Android app and run the model inference on your device.

💡**NOTE**: If you already have a trained TIMM model, feel free to jump straight into [Exporting to TorchScript](https://dicksonneoh.com/portfolio/timm_torchscript_flutter/#-exporting-to-torchscript) section.
{{< /notice >}}

Demo of the app 👇

![img](./vids/anim.gif)


<!-- You might wonder, do I need to learn ONNX? TensorRT? TFLite?

Maybe.

Learning each on of them takes time. Personally, I never had a very positive experience with exporting PyTorch models into ONNX.
It doesn't work every time. -->
<!-- I had to pull my hair over sleepless nights exporting to ONNX.
They are out of the PyTorch ecosystem. -->

<!-- But in this post I will show you solution that holds the best chances of working - TorchScript. -->
<!-- Integrated within the PyTorch ecosystem. -->

If that looks interesting, read on 👇

### 🌿 Dataset
We will be working with the Paddy Disease Classification [dataset](https://www.kaggle.com/competitions/paddy-disease-classification) from Kaggle. 

The dataset consist of `10,407` labeled images across ten classes (9 diseases and 1 normal):
1. `bacterial_leaf_blight`
2. `bacterial_leaf_streak`
3. `bacterial_panicle_blight`
4. `blast`
5. `brown_spot`
6. `dead_heart`
7. `downy_mildew`
8. `hispa`
9. `tungro`
10. `normal`

The task is to classify the paddy images into `1` of the `9` diseases or `normal`. 

Here's how the images look like.
{{< figure_resizing src="test_img.jpg" >}}

Next, I download the data locally and organize them in a folder structure. 
Here's the structure I have on my computer.
```tree
├── data
│   ├── test_images
│   └── train_images
│       ├── bacterial_leaf_blight 
│       ├── bacterial_leaf_streak 
│       ├── bacterial_panicle_blight 
│       ├── blast 
│       ├── brown_spot 
│       ├── dead_heart 
│       ├── downy_mildew 
│       ├── hispa 
│       ├── models
│       ├── normal 
│       └── tungro 
└── train
    └── train.ipynb
```
{{< notice note >}}
Descriptions of the folders:
+ `data/` - A folder to store train and test images.
+ `train/` - A folder to store training related files and notebooks.

View the full structure by browsing my GitHub [repo](https://github.com/dnth/timm-flutter-pytorch-lite-blogpost).
{{< /notice  >}}

{{< notice tip >}}
If you'd like to explore the dataset and excel in the competition, I'd encourage you to check out a series of Kaggle notebooks by Jeremy Howard.
+ [First Steps.](https://www.kaggle.com/code/jhoward/first-steps-road-to-the-top-part-1)
+ [Small Models.](https://www.kaggle.com/code/jhoward/small-models-road-to-the-top-part-2)
+ [Scaling Up.](https://www.kaggle.com/code/jhoward/scaling-up-road-to-the-top-part-3)
+ [Multi-target.](https://www.kaggle.com/code/jhoward/multi-target-road-to-the-top-part-4)

I've personally learned a lot from the notebooks. Part of the codes in the post is adapted from the notebooks.
{{< /notice  >}}

Now that we've got the data, let's see how to start building a model out of it

For that we need 👇

### 🥇 PyTorch Image Models
There are many libraries to model computer vision tasts but PyTorch Image Models or [TIMM](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://www.linkedin.com/in/wightmanr/) is arguably the most prominent one today.

The TIMM repository hosts hundreds of recent SOTA models maintained by Ross.
At this point (January 2023) we have 964 pretrained model on TIMM and increasing as we speak.

You can install TIMM by simply:
```bash
pip install timm
```

One line of code, and we'd have access to all models on TIMM!

With such a massive collection, it can be disorienting which model to start from.
Worry not, TIMM provides a function to search for model architectures a [wildcard](https://www.delftstack.com/howto/python/python-wildcard/).

Since we will be running the model on a mobile device, let's search for models that has the word `edge`.

```python
import timm
timm.list_models('*edge*')
```

This outputs all models that match the wildcard.
```bash
['cs3edgenet_x',
 'cs3se_edgenet_x',
 'edgenext_base',
 'edgenext_small',
 'edgenext_small_rw',
 'edgenext_x_small',
 'edgenext_xx_small']
```
With the right model name, you can start training.
The TIMM repo also provides various utility functions and training script. Feel free to use them.

In this post I'm going to show you an easy way to train a TIMM model using Fastai 👇


### 🏋️‍♀️ Training with Fastai
[Fastai](https://www.fast.ai/2020/02/13/fastai-A-Layered-API-for-Deep-Learning/) is a deep learning library which provides practitioners with high high-level components that can quickly provide SOTA results.
Under the hood Fastai uses PyTorch but it abstracts away the details and incorporates various best practices in training a model.

Install Fastai with:
```bash
pip install fastai
```

Since, we'd run our model on a mobile device, let's select the smallest model we got from the previous section - `edgenext_xx_small`.

Now let's use Fastai and quickly train the model. Let's import all the necessary packages with:
```python
from fastai.vision.all import *
```

Next, load the images into a `DataLoader`.

```python
trn_path = Path('../data/train_images')
dls = ImageDataLoaders.from_folder(trn_path, seed=316, 
                                   valid_pct=0.2, bs=128,
                                   item_tfms=[Resize((224, 224))], 
                                   batch_tfms=aug_transforms(min_scale=0.75))
```

{{< notice note >}}

Parameters for the `from_folder` method:

* `trn_path` -- A `Path` to the training images.
* `valid_pct` -- The percentage of dataset to allocate as the validation set.
* `bs` -- Batch size to use during training.
* `item_tfms` -- Transformation applied to each item.
* `batch_tfms` -- Random transformations applied to each batch to augment the dataset.


{{< /notice >}}

You can show a batch of the images loaded into the `DataLoader` with:

```python
dls.train.show_batch(max_n=8, nrows=2)
```

{{< figure_resizing src="show_batch.png" >}}

Next create a `Learner` object which combines the model and data into one object for training.

```python
learn = vision_learner(dls, 'edgenext_xx_small', metrics=accuracy).to_fp16()
```

Find the best learning rate.

```python
learn.lr_find()
```

{{< figure_resizing src="lr_find.png" >}}

Now train the model.

```python
learn.fine_tune(5, base_lr=1e-2, cbs=[ShowGraphCallback()])
```

{{< figure_resizing src="train.png" >}}


Optionally export the Learner.

```python
learn.export("../../train/export.pkl")
```

### 📀 Exporting to TorchScript
Now that we are done training the model, it's time we export the model in a form suitable on a mobile device.

We can do that easily with [TorchScript](https://pytorch.org/docs/stable/jit.html).
TorchScript is a way to create serializable and optimizable models from PyTorch code on 
a variety of platforms, including desktop and mobile devices, without requiring a Python runtime. 

With TorchScript, the model's code is converted into a static graph that can be optimized for faster performance, and then saved and loaded as a serialized representation of the model. This allows for deployment to a variety of platforms and acceleration with hardware such as GPUs, TPUs, and mobile devices.

<!-- {{% blockquote author="TorchScript Docs" %}}
TorchScript is a way to create serializable and optimizable models from PyTorch code.
{{% /blockquote %}} -->

All the models on TIMM can be exported to TorchScript with the following code snippet.

```python {linenos=table}
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

learn.model.cpu()
learn.model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(learn.model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("torchscript_edgenext_xx_small.pt")
```

&nbsp;

{{< notice note >}}
From the snippet above we need to specify a few things:
+ `Line 6`: The shape of the input image tensor.
+ `Line 9`: "torchscript_edgenext_xx_small.pt" is the name of the resulting TorchScript serialized model.
{{< /notice >}}


Once completed, you'll have a file `torchscript_edgenext_xx_small.pt` that can be ported to other devices for inference.
In this post, I will be porting it to Android using a framework known as Flutter. 

### 📲 Inference in Flutter
We can load the `torchscript_edgenext_xx_small.pt` and use if for inference.
To do so, we will use the [pytorch_lite](https://github.com/zezo357/pytorch_lite) Flutter package.
The `pytorch_lite` package supports image classification and detection with TorchScript.

The following code snippet shows a function to load our serialized model `torchscript_edgenext_xx_small.pt`.

```dart {linenos=table}
Future loadModel() async {
    String pathImageModel = "assets/models/torchscript_edgenext_xx_small.pt";
    try {
        _imageModel = await PytorchLite.loadClassificationModel(
            pathImageModel, 224, 224,
            labelPath: "assets/labels/label_classification_paddy.txt");
    } on PlatformException {
        print("only supported for Android");
    }
}
```

&nbsp;

{{< notice note >}}
From the snippet above we need to specify a few things:
+ `Line 2`: Path to the serialized model.
+ `Line 5`: The input image size - `224` by `224` pixels.
+ `Line 6`: A text file with labels associated with each class.

View the full code on my GitHub [repo](https://github.com/dnth/timm-flutter-pytorch-lite-blogpost/blob/main/flutter_app/lib/main.dart).

{{< /notice >}}


The following code snippet shows a function to run the inference.
```dart {linenos=table}
Future runClassification() async {
    objDetect = [];
    //pick a random image
    final PickedFile? image =
        await _picker.getImage(source: ImageSource.gallery);

    //get prediction
    _imagePrediction = await _imageModel!
        .getImagePrediction(await File(image!.path).readAsBytes());

    List<double?>? predictionList = await _imageModel!.getImagePredictionList(
      await File(image.path).readAsBytes(),
    );

    List<double?>? predictionListProbabilites =
        await _imageModel!.getImagePredictionListProbabilities(
      await File(image.path).readAsBytes(),
    );

    //Gettting the highest Probability
    double maxScoreProbability = double.negativeInfinity;
    double sumOfProbabilites = 0;
    int index = 0;
    for (int i = 0; i < predictionListProbabilites!.length; i++) {
      if (predictionListProbabilites[i]! > maxScoreProbability) {
        maxScoreProbability = predictionListProbabilites[i]!;
        sumOfProbabilites = sumOfProbabilites + predictionListProbabilites[i]!;
        index = i;
      }
    }
    _predictionConfidence = (maxScoreProbability * 100).toStringAsFixed(2);

  }
```
Those are the two important functions to load and run the TorchScript model.

The following screen capture shows the Flutter app in action. 
The clip runs in real-time and not sped up! 

{{< video src="vids/inference_edgenext_new.mp4" width="400px" loop="true" autoplay="true" muted="true">}}

The compiled `.apk` file is about **77MB** in size.
Install the pre-built `.apk` file on your Android phone [here](https://github.com/dnth/timm-flutter-pytorch-lite-blogpost/blob/main/app-release.apk?raw=true).

### 🙏 Comments & Feedback
I hope you've learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter/LinkedIn post or [drop me a message](https://dicksonneoh.com/contact/).
<!-- {{< tweet dicksonneoh7 1534395572022480896>}}


<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:6940225157286264834" height="2406" width="550" frameborder="0" allowfullscreen="" title="Embedded post"></iframe> -->

