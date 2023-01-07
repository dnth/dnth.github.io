---
title: "Fastdup: An Underrated Tool to Manage, Clean & Curate Visual Data at Scale on a Single CPU - For Free."
date: 2023-01-03T11:00:15+08:00
featureImage: images/portfolio/fastdup_manage_clean_curate/thumbnail.gif
postImage: images/portfolio/fastdup_manage_clean_curate/post_image.png
tags: ["Fastdup", "Fast.ai", "scene-classification"]
categories: ["data-validation", "object-classification", "modeling"]
toc: true
socialshare: true
description: "Manage, Clean & Curate Visual Data at Scale on a CPU! For free!"
images : 
- images/portfolio/fastdup_manage_clean_curate/post_image.png
---

{{< notice info >}}
This blog post is still a work in progress. If you require further clarifications before the contents are finalized, please get in touch with me [here](https://dicksonneoh.com/contact/), on [LinkedIn](https://www.linkedin.com/in/dickson-neoh/), or [Twitter](https://twitter.com/dicksonneoh7).
{{< /notice >}}

### ✅ Motivation

As a data scientist, you might be tempted to jump into modeling as soon as you can.
I mean, that's the fun part, right? 

But trust me, if you skip straight to modeling without taking the time to really 
understand the problem and analyze the data, you're setting yourself up for failure.

I've been there.

You might feel like a superstar, but you'll have with a model that doesn't work 🤦‍♂️.

{{< figure_resizing src="meme2.jpeg" caption="" >}}

A model can only be as good as the data it's trained on.
Bad data produce bad models.

**But how exactly do we inspect large volumes of images? And do it on a local computer quickly, for free?**

Sounds too good to be true eh?

It's not, with 👇

### ⚡ Fastdup

Fastdup is a data-cleaning tool that lets you manage, clean, and curate your images at scale.
It's incredibly easy to use and highly efficient. 

At first, I was skeptical. How could a single tool handle all my data cleaning and curation needs on a single CPU machine? Especially if the dataset is huge. But I was curious, so I decided to give it a try. And I have to say, I was pleasantly surprised.

Fastdup lets me clean my visual data with ease, freeing up valuable resources and time. 
But that's not all - it also had powerful curation features that helped me organize and prioritize my data, making it easier to find what I needed when I needed it.

Fastdup lets you find 👇

{{< figure_resizing src="features.png" caption="Fastdup superpowers. Source: Fastdup GitHub." link="https://github.com/visual-layer/fastdup" >}}


In short, Fastdup is 👇
* **Unsupervised**: fits any visual dataset.
* **Scalable** : handles 400M images on a single machine.
* **Efficient**: works on CPU (even on [Colab](https://colab.research.google.com/github/visualdatabase/fastdup/blob/main/examples/fastdup.ipynb) with only 2 CPU cores!).
* **Low Cost**: can process 12M images on a $1 cloud machine budget.

The best part? Fastdup is **free**.

It's easy to get started and use. 
I think it should in your toolbox if you're doing computer vision.
The [authors of Fastdup](https://www.visual-layer.com/) used it to uncover over **1.2M duplicates** and **104K data train/validation leaks** in the ImageNet-21K dataset.
Read more [here](https://medium.com/@amiralush/large-image-datasets-today-are-a-mess-e3ea4c9e8d22).

{{< notice tip >}}
⚡ By the end of this post, you will learn how to:

* Install Fastdup and run it on your local machine.
* Find duplicates and anomalies in your dataset.
* Identify wrong/confusing labels in your dataset. 
* Uncover data leaks in your dataset.
* Train a state-of-the-art model with [Fastai](https://www.fast.ai/).

**NOTE**: All codes are on my [Github repo](https://github.com/dnth/fastdup-manage-clean-curate-blogpost). Alternatively, you can [run this example in Colab](https://github.com/dnth/fastdup-manage-clean-curate-blogpost/blob/main/clean.ipynb).

{{< /notice >}}

If that looks interesting, let's dive in.

### 📖 Installation
To start, run:

```bash
pip install fastdup
```
I'm running version `fastdup==0.189` for this post.

### 🖼 Dataset
We will be using an openly available image classification [dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) from Intel.
The dataset contains 25,000 images (150 x 150 pixels) of natural scenes from around the world in six categories:
1. buildings
2. forest
3. glacier
4. mountain
5. sea
6. tree

{{< figure_resizing src="dataset_sample.png" caption="Samples from dataset." link="https://www.kaggle.com/datasets/puneet6060/intel-image-classification" >}}

### 🏋️‍♀️ Fastdup in Action: Discovering Data Issues

Once we have the data locally, let's organize the folder structure. Here's what I have on my machine.

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
Here's a brief description of each directory:
+ `data/` -- Folder to store all datasets.
+ `report/` -- Directory to save the output generated by Fastdup.

{{< /notice >}}

Next, all you have to do is run:

```python
import fastdup
fastdup.run(input_dir='scene_classification/data/train_set/', 
            work_dir="scene_classification/report/train/")
```

{{< notice note >}}
Parameters for the `run` method:
* `input_dir` -- Path to the folder containing images. In this example, we are checking the training dataset. Change this accordingly if you're running this on the validation/test set.
* `work_dir` -- **Optional**. Path to save the outputs from the run. If not specified, the output will be saved to the current directory.
{{< /notice >}}

Fastdup will run through all images in the folder to check for issues.
How long it takes depends on how powerful is your CPU. 
On my machine, with an Intel Core™ i9-11900 it takes **under 1 minute** to check through (approx. 25,000) images in the folder 🤯.

Once complete, you'll find a bunch of output files in the `work_dir` folder.
We can now visualize them.


{{< notice tip >}}
In this post, I'm only running on the `train_set` folder to illustrate what's possible. 

Feel free to repeat the steps for `valid_set` and `test_set` by pointing `input_dir` and `work_dir` to the appropriate folders. 

{{< /notice >}}

#### 🧑‍🤝‍🧑 Duplicates

First, let's see if there are duplicates in the `train_set`. Let's load the file and visualize them with:

```python
from IPython.display import HTML
fastdup.create_duplicates_gallery(similarity_file='scene_classification/report/train/similarity.csv'
                                  save_path='scene_classification/report/train/', 
                                  num_images=5)

HTML('scene_classification/report/train/similarity.html')
```
{{< notice note >}}

* `similarity_file` -- A `.csv` file with the computer similarity generated by the `run` method.
* `save_path` -- Path to save the visualization. Defaults `'./'`.
* `num_images` -- The max number of images to display. Defaults to `50`. For brevity, I've set it to `5`.

{{< /notice >}}

You'd see something like the following 👇

{{< include_html "./content/portfolio/fastdup_manage_clean_curate/similarity.html" >}}


{{< notice info >}}


Here, we can already spot a couple of issues in the `train_set`:

* On `row 1` and `row 2` of the table. We see that `19255.jpg` and `7872.jpg`
are **duplicates of the same class**. We know this by the `Distance` value of `1.0`. 
You can also see that they are exactly the same side-by-side. 

* On `row 0`, images `9769.jpg` and `7293.jpg` are exact copies but they exist in both the `buildings` and `street` folders!
The same can be seen on `row 3` and `row 4`. 
These are **duplicate images but labeled as different classes** and will end up confusing your model!

{{< /notice >}}

For the sake of simplicity I've only shown five rows, if you run the code, you'd find more!
Eliminating these images can already improve your model quite a bit.

Using Fastdup we can remove the duplicates quite easily.
```python
top_components = fastdup.find_top_components(work_dir="scene_classification_clean/report/")
fastdup.delete_components(top_components, None, how='one', dry_run=False)
```

That's how easy it is to find duplicate images in your dataset! Let's see if we can find more issues.

#### 🦄 Anomalies
Similar to duplicates, it's also easy to visualize anomalies in your dataset:

```python
fastdup.create_outliers_gallery(outliers_file='scene_classification/report/train/outliers.csv',            
                                save_path='scene_classification/report/train/', 
                                num_images=5)
HTML('scene_classification/report/train/outliers.html')
```
You'd see something like the following 👇

{{< include_html "./content/portfolio/fastdup_manage_clean_curate/outliers.html" >}}

{{< notice info >}}

What do we find here?

* Image `12723.jpg` in the top row is labeled as `glacier`, but it doesn't look like one to me. 
* Image `5610.jpg` doesn't look like a `forest`.

**NOTE**: Run the code snippet and increase the `num_images` parameter to see more anomalies. 
Also, repeat this with `valid_set`.
{{< /notice >}}

All the other images don't look too convincing to me either.
I guess you can evaluate the rest if they belong to the right classes as labeled.

Remove these outliers programmatically with:
```python
fastdup.delete_or_retag_stats_outliers(stats_file="scene_classification_clean/report/outliers.csv", 
                                       metric='distance', filename_col='from', 
                                       lower_threshold=0.6, dry_run=False)
```


#### 💆 Wrong or Confusing Labels
Other than duplicates and anomalies, one of my favorite capabilities of Fastdup is finding wrong or confusing labels.
Similar to previous sections, we can simply run:

```python
fastdup.create_similarity_gallery(similarity_file="scene_classification/report/train/similarity.csv", 
                                  save_path="scene_classification/report/train/", 
                                  get_label_func=lambda x: x.split('/')[-2], 
                                  num_images=5, max_width=180, slice='label_score', 
                                  descending=False)
HTML('./scene_classification/report/train/topk_similarity.html')
```

You'd see something like 👇

{{< include_html "./content/portfolio/fastdup_manage_clean_curate/topk_similarity.html" >}}

Under the hood, Fastdup finds images that are similar to one another at the embedding level but are assigned different labels.

{{< notice info >}}
What can we observe here?

* On the top row, we find that `3279.jpg` is labeled `forest` but looks very similar to `mountains`. 
* On the remaining rows, we see confusing labels between `glacier` and `mountains`.

{{< /notice >}}

It is important to address these confusing labels because if the training data contains confusing or incorrect labels, it can negatively impact the performance of the model.

At this point, you might want to invest some time to review and correct these wrong/confusing labels before training a model.


{{< notice tip >}}
You can repeat the steps to find duplicates, anomalies, and problematic labels for the `valid_set` and `test_set`. 
To do so, you'd have to call the `run` method again specifying the appropriate dataset folders.
{{< /notice >}}

Using Fastdup we can delete or retag these images.

```python
fastdup.delete_or_retag_stats_outliers(stats_file=df, metric='score', filename_col='from', lower_threshold=51, dry_run=False, how='delete')
```

#### 🚰 Data Leakage
In the [first section](#-duplicates) we tried finding duplicates within the `train_set`. We found a few duplicate images within the same folder.

In this section, we try to find duplicates from the `train_set` and `valid_set`.
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
* The `input_dir` and `test_dir` point to different locations. 

* The `test_dir` parameter should point to `valid_set` or `test_set` folder.
{{< /notice >}}


Running the codes we find 👇

{{< include_html "./content/portfolio/fastdup_manage_clean_curate/train_valid_similarity.html" >}}

From the visualization above, we find various duplicates from the `train_set` and `valid_set`.

{{< notice info >}}
Upon careful observation, you'd notice two obvious issues:

* Duplicate images in **different dataset** - On the top row, `21469.jpg` and `14341.jpg` are duplicates but they exist in `train_set` and `valid_set` respectively.

* Duplicate images with **different labels** - On the top row, `21469.jpg` is labeled as `glacier` in the `valid_set` and `mountain` in the `train_set`.

{{< /notice >}}

This is bad news. We just uncovered a **train-validation data leakage**! 

This is a common reason a model performs all too well during training and fails in production because the model might just 
memorize the training set without generalizing to unseen data.

{{< notice tip >}}
* A validation set consists of **representative** and **non-overlapping** samples from the train set and is used to evaluate models during training.
* Overlapping images in the train and validation set may lead to poor performance on new data.
* The way we craft our validation set is extremely important to ensure the model does not overfit. 
{{< /notice >}}

Spending time crafting your validation set though takes a little effort, but will pay off well in the future.

Rachel Thomas from [Fastai](https://www.fast.ai/) wrote a good piece on [how to craft
a good validation set](https://www.fast.ai/posts/2017-11-13-validation-sets.html).

<!-- ### 📖 Baseline Performance - Fastai
With the unmodified dataset let's model a quick model it using Fastai.

Using Fastai, you can create a reasonably decent model and train it with the best practices included.

View my training notebook [here](https://github.com/dnth/fastdup-blogpost/blob/main/train.ipynb).

```python {linenos=table}
from fastai.vision.all import *
path = Path('./scene_classification/data/seg_train/')
block = DataBlock(
            blocks=(ImageBlock, CategoryBlock), 
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            get_y=parent_label, item_tfms=[Resize(150)],
            batch_tfms=aug_transforms(mult=1.5, size=384, min_scale=0.75))
loaders = block.dataloaders(path)
learn = cnn_learner(loaders, resnet18, metrics=accuracy)
learn.fine_tune(5, base_lr=1e-3)
```
The above are all the codes you'll need to create a CNN model (resnet18) that performs >90% accuracy!

Confusion matrix. -->


<!-- ### 🎯 Optimized Performance - Fastdup + Fastai -->


### 🙏 Comments & Feedback

That's a wrap!

{{< notice tip >}}

In this post I've shown you how to: 

* **Install** Fastdup and run it on your local machine.
* Find **duplicate** and **anomalies** in your dataset.
* Identify **wrong/confusing labels** in your dataset. 
* Uncover **data leak** in your dataset.
* Train a state-of-the-art model with [Fastai](https://www.fast.ai/).
{{< /notice >}}

Additionally, I trained a deep learning classification model on the clean data using Fastai. View the training notebook [here](https://github.com/dnth/fastdup-manage-clean-curate-blogpost/blob/main/train_clean.ipynb). 
The performance on the validation set is approximately 94.9% comparable to the [winning solutions of the competition](https://medium.com/@afzalsayed96/1st-place-solution-for-intel-scene-classification-challenge-c95cf941f8ed).

I hope you've enjoyed and learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter/LinkedIn post or [drop me a message](https://dicksonneoh.com/contact/).
<!-- {{< tweet dicksonneoh7 1534395572022480896>}}


<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:6940225157286264834" height="2406" width="550" frameborder="0" allowfullscreen="" title="Embedded post"></iframe> -->

