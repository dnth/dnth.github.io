---
title: "fastdup: A Powerful Tool to Manage, Clean & Curate Visual Data at Scale on Your CPU - For Free."
date: 2023-01-03T11:00:15+08:00
featureImage: images/portfolio/fastdup_manage_clean_curate/thumbnail.gif
postImage: images/portfolio/fastdup_manage_clean_curate/post_image.gif
tags: ["fastdup", "natural-scene-classification", "intel"]
categories: ["data-cleaning", "image-classification"]
toc: true
socialshare: true
description: "Learn how to use fastdup to find image duplicates, anomalies, wrong labels and data leakage. Say goodbye to cluttered folders."
images : 
- images/portfolio/fastdup_manage_clean_curate/post_image.gif
---

⏳ **Last Updated**: March 27, 2023.
### ✅ Motivation

As a data scientist, you might be tempted to jump into modeling as soon as you can.
I mean, that's the fun part, right? 

But trust me, if you skip straight to modeling without taking the time to really 
understand the problem and analyze the data, you're setting yourself up for failure.

I've been there.

You might feel like a superstar, but you'll have with a model that doesn't work 🤦‍♂️.

{{< figure_resizing src="meme2.jpeg" caption="" >}}

<!-- A model can only be as good as the data it's trained on.
Bad data produce bad models. -->

But how do we even begin inspecting large datasets of images effectively and efficiently? And can we really do it on a local computer quickly, for free?

Sounds too good to be true eh?

It's not, with 👇

### ⚡ fastdup

fastdup is a tool that let us gain insights from a large image/video collection. 
You can manage, clean, and curate your images at scale on your local machine with a single CPU.
It's incredibly easy to use and highly efficient. 

At first, I was skeptical. How could a single tool handle my data cleaning and curation needs on a single CPU machine, especially if the dataset is huge? But I was curious, so I decided to give it a try. 

And I have to say, I was pleasantly surprised.

fastdup lets me clean my visual data with ease, freeing up valuable resources and time. 

Here are some superpowers you get with fastdup.
It lets you identify:

{{< figure_resizing src="features.png" caption="fastdup superpowers. Source: fastdup GitHub." link="https://github.com/visual-layer/fastdup" >}}


In short, fastdup is 👇
* **Fast**: Efficient C++ engine processes up to 7000 images in less than 3 minutes with a 2-core CPU (Google Colab).
* **Scalable**: Handles up to 400M images on a single CPU machine.
* **Unsupervised**: Runs on unlabeled (or labeled) image/video data.
* **Cost**: Basic functions are free to use. Process up to 12M images on a $1 cloud machine budget. 

The best part? fastdup is **free**. 

It's easy to get started and use. 
The [authors of fastdup](https://www.visual-layer.com/) even used it to uncover over **1.2M duplicates** and **104K data train/validation leaks** in the ImageNet-21K dataset [here](https://medium.com/@amiralush/large-image-datasets-today-are-a-mess-e3ea4c9e8d22).

{{< notice tip >}}
⚡ By the end of this post, you will learn how to:

* **Install** fastdup and run it on your local machine.
* Find **duplicate** and **anomalies** in your dataset.
* Identify **wrong/confusing labels** in your dataset. 
* Uncover **data leak** in your dataset.

📝 **NOTE**: All codes used in the post are on my [Github repo](https://github.com/dnth/fastdup-manage-clean-curate-blogpost). Alternatively, you can [run this example in Colab](https://github.com/dnth/fastdup-manage-clean-curate-blogpost/blob/main/clean.ipynb).

{{< /notice >}}

If that looks interesting, let's dive in.

### 📖 Installation
To start, run:

```bash
pip install fastdup==0.909
```
I'm using version 0.909 for this post but, feel free to use the latest version available if they are compatible.

### 🖼 Dataset
I will be using an openly available image classification [dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) from Intel.
The dataset contains 25,000 images (150 x 150 pixels) of natural scenes from around the world in 6 categories:
1. `buildings`
2. `forest`
3. `glacier`
4. `mountain`
5. `sea`
6. `tree`

{{< figure_resizing src="dataset_sample.png" caption="Samples from dataset." link="https://www.kaggle.com/datasets/puneet6060/intel-image-classification" >}}

{{< notice tip >}}
I encourage you to pick a dataset of your choice in running this example. You can find some inspiration [here](https://paperswithcode.com/datasets?task=image-classification).
{{< /notice >}}


### 🏋️‍♀️ fastdup in Action: Discovering Data Issues
Next, download the data locally and organize them in a folder structure. 
Here's the structure I have on my computer.
```tree
├── scene_classification
   ├── data
   │   ├── train_set
   │   |   ├── buildings
   │   |   |   ├── image1.jpg
   │   |   |   ├── image2.jpg
   │   |   |   ├── ...
   │   |   ├── mountain
   │   |   |   ├── image10.jpg
   │   |   |   ├── image11.jpg
   │   |   |   ├── ...
   |   ├── valid_set
   |   |   ├── buildings
   |   |   |   ├── image100.jpg
   │   |   |   ├── ...
   |   └── test_set
   └── report
        ├── train
        ├── train_valid
        └── valid
```

{{< notice note >}}
Description of folders:
+ `data/` -- Folder to store all datasets.
+ `report/` -- Directory to save the output generated by fastdup.

📝 **NOTE**: For simplicity, I've also included the datasets in my [Github repo](https://github.com/dnth/fastdup-manage-clean-curate-blogpost).
{{< /notice >}}

To start checking through the images, create a Jupyter notebook and run:

```python
work_dir = "scene_classification/report/"
images_dir = "scene_classification/data/"

fd = fastdup.create(work_dir, images_dir)
fd.run()
```

{{< notice note >}}
Parameters for the `run` method:
* `work_dir` -- Path to save the output artifacts from the run.
* `input_dir` -- Path to the folder containing images.


**📝 NOTE**: More info on other parameters [here](https://visual-layer.readme.io/docs/v1-api#fastdup.engine.Fastdup.run).
{{< /notice >}}

fastdup will run through all images in the folder to check for issues.
How long it takes depends on how powerful is your CPU. 
On my machine, with an Intel Core™ i9-11900 it takes about **1 minute** to check through (approx. 25,000) images in the folder 🤯.

<!-- {{< notice tip >}}
In this post, I'm only running on the `train_set` folder to illustrate what's possible. 

Feel free to repeat the steps for `valid_set` and `test_set` by pointing `input_dir` and `work_dir` to the appropriate folders. 

{{< /notice >}} -->

Once complete, a run summary will be printed out.

```bash
Found a total of 118 fully identical images (d>0.990), which are 0.16 %
Found a total of 108 nearly identical images(d>0.980), which are 0.15 %
Found a total of 11908 above threshold images (d>0.900), which are 16.31 %
Found a total of 2433 outlier images         (d<0.050), which are 3.33 %

```

You'll also find a bunch of output files in the `work_dir` folder.
We can now visualize them accordingly.

The upcoming sections show how you can visualize [duplicates](#-duplicates), [anomalies](#-anomalies), [confusing labels](#-wrong-or-confusing-labels) 
and [data leakage](#-data-leakage). 

#### 🧑‍🤝‍🧑 Duplicates

First, let's see if there are duplicates in the dataset with:

```python
fd.vis.duplicates_gallery()
```
{{< notice note >}}
Other parameters for `duplicates_gallery` method:
* `save_path` -- Path to save the visualization. Defaults `work_dir`.
* `num_images` -- The max number of images to display. Defaults to `20`.

**📝 NOTE**: More info on other parameters [here](https://visual-layer.readme.io/docs/v1-api#fastdup.fastdup_galleries.FastdupVisualizer.duplicates_gallery).

{{< /notice >}}

You'd see something like the following 👇

{{< figure_resizing src="duplicates_report.png" caption="" link="./duplicates_report.png">}}


<!-- {{< notice info >}}

We can already spot a few issues in the `train_set`:

* On `row 1`, note that `19255.jpg` and `7872.jpg`
are **duplicates of the same class**. We know this by the `Distance` value of `1.0`. 
You can also see that they are exactly the same side-by-side. The same with `row 2`.

* On `row 0`, images `9769.jpg` and `7293.jpg` are exact copies but they exist in both the `buildings` and `street` folders!
The same can be seen on `row 3` and `row 4`. 
These are **duplicate images but labeled as different classes** and will end up confusing your model!

{{< /notice >}} -->

Here, we can already spot a few issues in our dataset. As shown below, `10234.jpg` and `7654.jpg` are exact duplicates.
We know that through the `Distance` score of `1.0`.

{{< figure_resizing src="dup_1.png" caption="" link="./dup_1.png">}}

But that's not the only problem.
They are labeled differently! One if labeled `glacier` and the other `mountain`.
If you look further there are a bunch of other duplicates too.

In this example both the duplicates belong to the `train_set` (see the path of the images above).
In the event they do not belong to the same set, then we have 👇

#### 🚰 Data Leakage
In machine learning [data leakage](https://insidebigdata.com/2014/11/26/ask-data-scientist-data-leakage/) is when data from outside the training dataset is used to train the model. Read more about data leakage in machine learning [here](https://machinelearningmastery.com/data-leakage-machine-learning/).

Can you spot any data leakage in the gallery above?

{{< figure_resizing src="data_leakage.png" caption="" link="./data_leakage.png">}}

The gallery above shows that we indeed discovered a data leakage:
+ **Train-validation leak** - Image from the training set is found in the validation set.
+ **Train-test leak** - Image from the training set is found in the test set.

If you train a model on this data, you'd get a model that performs extremely well on the test set.
Along the process, you also convinced yourself that the model is robust. When in reality, it's not.

This is why models fail in production. 


It's because the model might just memorize the training set without generalizing to unseen data. 
That's why it's important to make sure the training and validation/test sets do not contain duplicates!

{{< notice tip >}}
* A validation set consists of **representative** and **non-overlapping** samples from the train set and is used to evaluate models during training.
* Overlapping images in the train and validation set may lead to poor performance on new data.
* The way we craft our validation set is extremely important to ensure the model does not overfit. 
{{< /notice >}}

Spending time crafting your validation set takes a little effort, but will pay off well in the future.
Rachel Thomas from [Fastai](https://www.fast.ai/) wrote a good piece on [how to craft
a good validation set](https://www.fast.ai/posts/2017-11-13-validation-sets.html).

<!-- Duplicate images do not provide value to your model, they take up hard drive space and increase your training time.
Eliminating these images improves your model performance, and reduces cloud billing costs for training and storage.

Plus, you save valuable time (and sleepless nights 🤷‍♂️) to train and troubleshoot your models down the pipeline. 

You can choose to remove the images by hand (e.g. going through them one by one and hitting the delete key on your keyboard.) There are cases you might want to do so.  -->

<!-- {{< notice warning >}}
The following code will **delete all duplicate images** from your folder. I recommend setting `dry_run=True` to see which files will be deleted.

📝 **NOTE**: Checkout the [fastdup documentation](https://visual-layer.github.io/fastdup/#fastdup.delete_components) to learn more about the parameters you can tweak.
{{< /notice >}}

```python
top_components = fastdup.find_top_components(work_dir="scene_classification_clean/report/")
fastdup.delete_components(top_components, dry_run=False)
```

In fastdup, a **component** is a **cluster** of similar images.
The snippet above removes duplicates of the same images (from the top cluster) ensuring you only have one copy of the image in your dataset. -->


That's how easy it is to find duplicate images and remove them from your dataset! 
Let's see if we can find more issues.

#### 🦄 Anomalies
Similar to duplicates, it's easy to visualize anomalies in your dataset:

```python
fd.vis.outliers_gallery()
```
You'd see something like the following 👇

{{< figure_resizing src="outliers_report.png" caption="" link="./outliers_report.png">}}

<!-- {{< notice info >}}

What do we find here?

* Image `12723.jpg` in the top row is labeled as `glacier`, but it doesn't look like one to me. 
* Image `5610.jpg` doesn't look like a `forest`.

📝 **NOTE**: Run the code snippet and increase the `num_images` parameter to see more anomalies. 
Also, repeat this with `valid_set` and see if there are more.
{{< /notice >}} -->

Well, what do we find here? The image below is certainly a mistake. 

It's not a broken image (it's still a valid image file) but there's no useful information on the image for it to be in the `test_set`.

{{< figure_resizing src="anomaly_1.png" caption="" link="./anomaly_1.png">}}

This type of image is common in [large dataset such as LAION and ImageNet](https://medium.com/@amiralush/large-image-datasets-today-are-a-mess-e3ea4c9e8d22).




All the other images above don't look too convincing to me either. Take a look at the images labeled as `forest` and `glacier` below. 

{{< figure_resizing src="anomaly_2.png" caption="" link="./anomaly_2.png">}}
{{< figure_resizing src="anomaly_3.png" caption="" link="./anomaly_3.png">}}

{{< notice tip>}}
Note that the lower the `Distance` value, the more likely it will be an outlier.
{{< /notice >}}


I guess you can evaluate the rest if they belong to the right classes as labeled. 
<!-- Now let's see how we can programmatically remove them. -->

<!-- {{< notice warning >}}
The following code will **delete all outliers** from your folder. I recommend setting `dry_run=True` to see which files will be deleted.

📝 **NOTE**: Checkout the [fastdup documentation](https://visual-layer.github.io/fastdup/#fastdup.delete_or_retag_stats_outliers) to learn more about the function parameters.
{{< /notice >}}

```python
fastdup.delete_or_retag_stats_outliers(stats_file="scene_classification_clean/report/outliers.csv", 
                                       metric='distance', filename_col='from', 
                                       lower_threshold=0.6, dry_run=False)
```
The above command removes all images with the `distance` value of `0.6` or below.

What value you pick for the `lower_threshold` will depend on the dataset. In this example, I notice that as `distance` go higher than `0.6`, the images look less like outliers. 

This isn't a foolproof solution, but it should remove the bulk of anomalies present in your dataset. -->

#### 💆 Wrong or Confusing Labels
One of my favorite capabilities of fastdup is finding wrong or confusing labels.
Similar to previous sections, we can simply run:

```python
fd.vis.similarity_gallery()
```

<!-- {{< notice note >}}
In case the dataset is labeled, you can specify the label using the function `get_label_func`. 

📝 **NOTE**: Check out the [fastdup documentation](https://visual-layer.github.io/fastdup/#fastdup.create_similarity_gallery) for parameters description.
{{< /notice >}} -->

You'd see something like 👇

{{< figure_resizing src="similarity_report.png" caption="" link="./similarity_report.png">}}

That looks like a lot of information. Let's break it down a little.

What's happening here is that, under the hood, fastdup finds images that are similar to one another at the embedding level but are assigned different labels.

For instance, the `glacier` image below is a duplicate of another image labeled `mountain` and another image of `glacier`. 

{{< figure_resizing src="similarity_1.png" caption="" link="./similarity_1.png">}}

It is important to address these confusing labels because if the training data contains confusing or incorrect labels, it can negatively impact the performance of the model.

<!-- A `score` metric is computed to reflect how similar the query image to the most similar images in terms of class label.

A **high** `score` means the query image looks similar to other images in the same class. Conversely, a **low** `score` indicates the query image is similar
to images from other classes.

{{< notice info >}}
What can we see in the table above?

* On the top row, we find that `3279.jpg` is labeled `forest` but looks very similar to `mountains`. 
* On the remaining rows, we see confusing labels between `glacier` and `mountains`.

{{< /notice >}}



{{< notice tip >}}
You can repeat the steps to find duplicates, anomalies, and problematic labels for the `valid_set` and `test_set`. 
To do so, you'd have to call the `run` method again specifying the appropriate dataset folders.
{{< /notice >}}

Using fastdup we can delete or retag these images.

{{< notice warning >}}
The following code will **delete all images with wrong labels** from your folder. I recommend setting `dry_run=True` to see which files will be deleted.

📝 **NOTE**: Checkout the [fastdup documentation](https://visual-layer.github.io/fastdup/#fastdup.delete_or_retag_stats_outliers) to learn more about the parameters.
{{< /notice >}}

```python
fastdup.delete_or_retag_stats_outliers(stats_file=df, 
                                       metric='score', 
                                       filename_col='from', 
                                       lower_threshold=51, 
                                       dry_run=False)
```

{{< notice note >}}
Parameters for `delete_or_retag_stats_outliers`:
* `stats_file` -- The output `DataFrame` from `create_similarity_gallery` method.
* `lower_threshold` -- The lower threshold value at which the images are deleted. In the above snippet, anything **lower** than a score of `51` gets **deleted**. The score is in the range 0-100 where 100 means this image is similar only to images from the same class label.
A score 0 means this image is only similar to images from other class labels.

📝 **NOTE**: Checkout the [fastdup documentation](https://visual-layer.github.io/fastdup/#fastdup.delete_or_retag_stats_outliers) to learn more about the parameters.

{{< /notice >}} -->

<!-- #### 🚰 Data Leakage
In the [Duplicates section](#-duplicates) above, we tried finding duplicates within the `train_set`. We found a few duplicate images within the same folder.

In this section, we check for duplicate images that exist in the train and validation dataset.
Technically, this should not happen. But let's find out.

We'd have to call the `run` method again and specify an additional parameter `test_dir`.

```python
import fastdup
fastdup.run(input_dir='scene_classification/data/train_set/', 
            work_dir="scene_classification/report/train_test/", 
            test_dir='scene_classification/data/valid_set/')

fastdup.create_duplicates_gallery(similarity_file='scene_classification/report/train_test/similarity.csv', 
                                  save_path='scene_classification/report/train_test/', num_images=5)
HTML('scene_classification/report/train_test/similarity.html')
```

{{< notice note >}}
Parameters for the in `run` method:
* The `input_dir` and `test_dir` point to different locations. 

* The `test_dir` parameter should point to `valid_set` or `test_set` folder.
{{< /notice >}}


Running the codes we find 👇

{{< include_html "./content/portfolio/fastdup_manage_clean_curate/train_valid_similarity.html" >}}

Note the **From** and **To** columns above now point `valid_set` and `train_set`.

{{< notice info >}}
From the table above, we find the following issues:

* Duplicate images in **different dataset** - On the top row, `21469.jpg` and `14341.jpg` are duplicates but they exist in `train_set` and `valid_set` respectively.

* Duplicate images with **different labels** - On the top row, `21469.jpg` is labeled as `glacier` in the `valid_set` and `mountain` in the `train_set`.

{{< /notice >}}
 -->


### 🙏 Comments & Feedback

That's a wrap!

{{< notice tip >}}

In this post I've shown you how to: 

* **Install** fastdup and run it on your local machine.
* Find **duplicate** and **anomalies** in your dataset.
* Identify **wrong/confusing labels** in your dataset. 
* Uncover **data leak** in your dataset.
{{< /notice >}}

By using fastdup and cleaning your dataset, you saved yourself:
* Unnecessary labeling cost.
* Long computation/training time.
* Headaches from debugging model predictions due to problems in data.

I believe fastdup is one of the easiest tools to get started for data cleaning. 
It’s a low-hanging fruit and ought to be in your toolkit if you’re working with image datasets.

If you're interested to learn more, I've trained a deep learning classification model on the clean version of the data using [Fastai](https://www.fast.ai/). View the training notebook [here](https://github.com/dnth/fastdup-manage-clean-curate-blogpost/blob/main/train_clean.ipynb). 
The accuracy on the validation set is approximately **94.9%** - comparable to the [winning solutions of the competition](https://medium.com/@afzalsayed96/1st-place-solution-for-intel-scene-classification-challenge-c95cf941f8ed) (96.48% with ensembling).

I hope you've enjoyed and learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter/LinkedIn post or [drop me a message](https://dicksonneoh.com/contact/).

{{< tweet dicksonneoh7 1618622445581393920>}}

<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:activity:7024313080490721280/" height="1760" width="550" frameborder="0" allowfullscreen="" title="Embedded post"></iframe>

