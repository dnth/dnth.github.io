---
title: "Fastdup: A Free, Underated Tool to Manage, Clean & Curate Visual Data at Scale on a Single CPU"
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

### ✅ Motivation
As an ML practitioner, you might be tempted to jump into modeling as soon as you can.
After all that is the sexiest part or the entire workflow.

But jumping straight into modeling without first spending time with the problem and data
is a perfect recipe for failure.

A model can only be as good as the data it's trained on.
Bad data produces bad model.

{{< figure_resizing src="meme.jpg" caption="The consequence of modeling over bad data. The model works but in a wrong way." >}}

But then how do you check your data? There are a plethora of data validation tools out there but they require some learning curve and time investement in the beginning.

Well, except Fastdup.

Fastdup is a data validation tool that let's you manage, clean and curate your images at scale.
It's easy to get started and use. This should be everyone's first step before diving into modeling.

{{< notice tip >}}
By the end of this post, you will learn how to:

* Install Fastdup and run it on your own dataset.
* Find dataset issues like duplicates, anomalies, wrong labels and train-test leak.
* Train a state-of-the-art classifier with Fastai library.

{{< /notice >}}

### ⚡ Fastdup
Okay first, what in the world is Fastdup?

{{% blockquote %}}
fastdup is a tool for gaining insights from a large image/video collection. It can find anomalies, duplicate and near duplicate images/videos, clusters of similarity, learn the normal behavior and temporal interactions between images/videos. It can be used for smart subsampling of a higher quality dataset, outlier removal, novelty detection of new information to be sent for tagging.{{% /blockquote %}}

Fastdup is:
* Unsupervised: fits any dataset
* Scalable : handles 400M images on a single machine
* Efficient: works on CPU only
* Low Cost: can process 12M images on a $1 cloud machine budget

From the authors of GraphLab and Turi Create.
Link to articles and other works.

{{< figure_resizing src="features.png" caption="Fastdup superpowers. Source: Gitub">}}

### 📖 Installation
To start run 

```bash
pip install fastdup
```

### 🖼 Dataset
We will be using an openly available image classification [dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) from Intel.

The dataset contains 25,000 images (150 x 150 pixels) of natural scenes from around the world in six categories:
* buildings
* forest
* glacier
* mountain
* sea
* tree

At this point it's tempting to start modeling right away.
Well let's that now - not for the reason you think, but to get a baseline model performance.


### 📖 Baseline Performance - Fastai
The easiest way to start modeling right away is by using Fastai.
With just a few lines of code you can create a reasonably recent model and train it with the best practices included.

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

Confusion matrix.

### 🏋️‍♀️ Fastdup in Action: Discovering Data Issues
So is it time to rejoice and call it a day?

The performance metric can be misleading.
Now let's see what underling issue with this.

We will be using Fastdup to discover the problems. 

It's so simple you'll only need to run one line of code.

```python
import fastdup
fastdup.run(input_dir='scene_classification/data/seg_train/', work_dir="scene_classification/report/train/")
```

View the notebook [here](https://github.com/dnth/fastdup-blogpost/blob/main/scene_train.ipynb).


#### 🧑‍🤝‍🧑 Duplicates
```python
from IPython.display import HTML
fastdup.create_duplicates_gallery('scene_classification/report/train/similarity.csv', save_path='scene_classification/report/train/', num_images=20, max_width=400)
HTML('scene_classification/report/train/similarity.html')
```

#### 🦄 Anomalies

```python
fastdup.create_outliers_gallery('scene_classification/report/train/outliers.csv', save_path='scene_classification/report/train/', num_images=10, max_width=400)
HTML('scene_classification/report/train/outliers.html')
```

#### 💆 Wrong or Confusing Labels

```python
fastdup.create_similarity_gallery("scene_classification/report/train/similarity.csv", save_path="scene_classification/report/train/", get_label_func=my_label_func, num_images=100,
                             get_reformat_filename_func=lambda x: os.path.basename(x), max_width=180, slice='label_score', descending=False)

HTML('./scene_classification/report/train/topk_similarity.html')
```

#### 🚰 Train-Test Leak
Find if there are duplciates in the train and test dataset.

```python
fastdup.create_duplicates_gallery('scene_classification/report/train_test/similarity.csv', save_path='scene_classification/report/train_test/', num_images=20, max_width=400)
HTML('scene_classification/report/train_test/similarity.html')
```

Yes there are duplicates found!
This means that the model might just memorize the data from the train set to do well on the test set.

### 🎯 Optimized Performance - Fastdup + Fastai


### 🙏 Comments & Feedback
Using fastdup with fastai lets you iterate quickly.
Fastdup lets you quickly check for data problems. 
Fastai lets you quickly train a model and validate.


I hope you've learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter/LinkedIn post or [drop me a message](https://dicksonneoh.com/contact/).
<!-- {{< tweet dicksonneoh7 1534395572022480896>}}


<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:6940225157286264834" height="2406" width="550" frameborder="0" allowfullscreen="" title="Embedded post"></iframe> -->

