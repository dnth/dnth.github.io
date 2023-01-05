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

As a computer vision practitioner, you might be tempted to jump into modeling as soon as you can.
After all that is the sexiest part of the entire workflow.

But, jumping straight into modeling without first spending time with the problem and data
is a perfect recipe for failure.

{{< figure_resizing src="meme2.jpeg" caption="" >}}

You can spend hours modeling only to find the model "works" but on the data.
Or worse, the model works, but is secretly failing - because of the data.
A model can only be as good as the data it's trained on.
Bad data produces bad model.

**But how exactly do you check your data? In computer vision the number of images can be huge. Can you do it quickly on your computer?**

Yes! With 👇

### ⚡ Fastdup

Fastdup is a data cleaning tool that let's you manage, clean and curate your images at scale.
It's incredibly easy to use and highly efficient. 

At first, I was skeptical. How could a single tool handle all my data cleaning and curation needs on a single CPU machine? Especially if the dataset is huge. But I was curious, so I decided to give it a try. And I have to say, I was pleasantly surprised.

Fastdup lets me clean my visual data with ease, freeing up valuable resources and time. 
But that's not all - it also had powerful curation features that helped me organize and prioritize my data, making it easier to find what I needed when I needed it.

Fastdup let's you find 👇

{{< figure_resizing src="features.png" caption="Fastdup superpowers. Source: Fastdup GitHub." link="https://github.com/visual-layer/fastdup" >}}


In short, Fastdup is 👇
* **Unsupervised**: fits any visual dataset.
* **Scalable** : handles 400M images on a single machine.
* **Efficient**: works on CPU (even on [Colab](https://colab.research.google.com/github/visualdatabase/fastdup/blob/main/examples/fastdup.ipynb) with only 2 CPU cores!).
* **Low Cost**: can process 12M images on a $1 cloud machine budget.

The best part? Fastdup is free.

It's easy to get started and use. 
I think it should in your toolbox if you're doing computer vision.

{{< notice tip >}}
By the end of this post, you will learn how to:

* Install Fastdup and run it on your own dataset on your local machine.
* Find dataset issues like duplicates, anomalies, wrong labels and train-test leak.
* Train a state-of-the-art classifier with Fastai library and compare the performance gain.
{{< /notice >}}

### 📖 Installation
To start run 

```bash
pip install fastdup
```

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

Once we have the data locally, let's organize the folder structure.

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
        └── test
```

{{< notice note >}}
+ `data/train_set/` - Where all images are placed.
+ `data/valid_set/` - Where all validation images are placed.
+ `data/test_set/` - Where all test images are placed.
+ `report/` - Where the reports from Fastdup are kept.

**NOTE**: You can explore further into the folder structure on my [Github repo](https://github.com/dnth/fastdup-manage-clean-curate-blogpost). Alternatively you can [run this example in Colab](https://github.com/dnth/fastdup-blogpost/blob/main/scene_train.ipynb).

{{< /notice >}}

Next, all you have to do is run:

```python
import fastdup
fastdup.run(input_dir='scene_classification/data/train_set/', 
            work_dir="scene_classification/report/train/")
```

{{< notice note >}}
* `input_dir` -- Path to the images folder. In this example we are checking the training dataset. Change this accordingly if you're running this on the validation/test set.
* `work_dir` -- Path to save the Fastdup reports.
{{< /notice >}}

Fastdup will run through all the images in the folder to check for issues.
How long it takes depends on how powerful is your CPU. 
On my machine with an Intel Core™ i9-11900 it takes **under 1 minute** to check through (approx. 25,000) images in the folder 🤯.

Once it's done running, you can view the reports generated.

#### 🧑‍🤝‍🧑 Duplicates
First, let's see if there are duplicates in the `train_set`.

Run:
```python
from IPython.display import HTML
fastdup.create_duplicates_gallery(similarity_file='scene_classification/report/train/similarity.csv'
                                  save_path='scene_classification/report/train/', 
                                  num_images=5)

HTML('scene_classification/report/train/similarity.html')
```

You'd see something like the following 👇

{{< include_html "./content/portfolio/fastdup_manage_clean_curate/similarity.html" >}}

We can already spot a couple of issues here in the `train_set`.

Firstly, in `row 1` and `row 2` of the table. We see that `19255.jpg` and `7872.jpg`
are **duplicates of the same class**. We know this by the `Distance` value of `1.0`. 
You can also see that they are exactly the same side-by-side. 

Next, here's a fun one, take a look at `row 0`.
Image `9769.jpg` and `7293.jpg` are exact copies but they exist in both the `buildings` and `street` folders!
The same can be seen on `row 3` and `row 4`. 
These are **duplicate images but labeled as different classes** and will end up confusing your model.

For the sake of simplicity I've only shown five rows, if you run the code, you'd find more!
Eliminating these images can already improve your model quite a bit.

<!-- {{< figure_resizing src="dup_1.png" caption="Fastdup superpowers. Source: Fastdup GitHub." >}} -->


#### 🦄 Anomalies
Next, let's take a look at the anomalies found in the `train_set`.

Run:
```python
fastdup.create_outliers_gallery(outliers_file='scene_classification/report/train/outliers.csv',            
                                save_path='scene_classification/report/train/', 
                                num_images=5)
HTML('scene_classification/report/train/outliers.html')
```
You'd see something like the following 👇

{{< include_html "./content/portfolio/fastdup_manage_clean_curate/outliers.html" >}}

What do we find here?

Image `12723.jpg` in the first row is labeled as `glacier`, but it doesn't look like one to me. I guess you can evaluate the rest if they belong to the right classes as labeled.

Again, I'm not showing the full list of anomalies here for brevity.
Run the code and you'll find more.


#### 💆 Wrong or Confusing Labels
We already found wrong labels by finding duplicates. Now 

```python
fastdup.create_similarity_gallery(similarity_file="scene_classification/report/train/similarity.csv",
                                  save_path="scene_classification/report/train/", get_label_func=my_label_func, num_images=5,
                                  get_reformat_filename_func=lambda x: os.path.basename(x), slice='label_score', descending=False)

HTML('./scene_classification/report/train/topk_similarity.html')
```

You'd see something like 👇

{{< include_html "./content/portfolio/fastdup_manage_clean_curate/topk_similarity.html" >}}



#### 🚰 Train-Test Leak
Find if there are duplciates in the train and test dataset.

```python
fastdup.create_duplicates_gallery('scene_classification/report/train_test/similarity.csv', save_path='scene_classification/report/train_test/', num_images=20, max_width=400)
HTML('scene_classification/report/train_test/similarity.html')
```

Yes there are duplicates found!
This means that the model might just memorize the data from the train set to do well on the test set.



### 📖 Baseline Performance - Fastai
With the unmodified dataset let's model a quickly model it using Fastai.

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

Confusion matrix.


### 🎯 Optimized Performance - Fastdup + Fastai


### 🙏 Comments & Feedback
Using fastdup with fastai lets you iterate quickly.
Fastdup lets you quickly check for data problems. 
Fastai lets you quickly train a model and validate.


I hope you've learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter/LinkedIn post or [drop me a message](https://dicksonneoh.com/contact/).
<!-- {{< tweet dicksonneoh7 1534395572022480896>}}


<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:6940225157286264834" height="2406" width="550" frameborder="0" allowfullscreen="" title="Embedded post"></iframe> -->

