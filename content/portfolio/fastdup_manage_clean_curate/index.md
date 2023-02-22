---
title: "Fastdup: A Powerful Tool to Manage, Clean & Curate Visual Data at Scale on Your CPU - For Free."
date: 2023-01-03T11:00:15+08:00
featureImage: images/portfolio/fastdup_manage_clean_curate/thumbnail.gif
postImage: images/portfolio/fastdup_manage_clean_curate/post_image.gif
tags: ["Fastdup", "natural-scene-classification", "intel"]
categories: ["data-cleaning", "image-classification"]
toc: true
socialshare: true
description: "Learn how to use Fastdup to clean and improve your visual data. Say goodbye to cluttered folders."
images : 
- images/portfolio/fastdup_manage_clean_curate/post_image.gif
---

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

### ⚡ Fastdup

Fastdup is a tool that let us gain insights from a large image/video collection. 
You can manage, clean, and curate your images at scale on your local machine with a single CPU.
It's incredibly easy to use and highly efficient. 

At first, I was skeptical. How could a single tool handle my data cleaning and curation needs on a single CPU machine, especially if the dataset is huge? But I was curious, so I decided to give it a try. 

And I have to say, I was pleasantly surprised.

Fastdup lets me clean my visual data with ease, freeing up valuable resources and time. 

Here are some superpowers you get with Fastdup.
It lets you identify:

{{< figure_resizing src="features.png" caption="Fastdup superpowers. Source: Fastdup GitHub." link="https://github.com/visual-layer/fastdup" >}}


In short, Fastdup is 👇
* **Unsupervised**: fits any visual dataset.
* **Scalable**: handles 400M images on a single machine.
* **Efficient**: works on CPU (even on Google Colab with only 2 CPU cores!).
* **Low Cost**: can process 12M images on a $1 cloud machine budget.

The best part? Fastdup is **free**. 

It's easy to get started and use. 
The [authors of Fastdup](https://www.visual-layer.com/) even used it to uncover over **1.2M duplicates** and **104K data train/validation leaks** in the ImageNet-21K dataset [here](https://medium.com/@amiralush/large-image-datasets-today-are-a-mess-e3ea4c9e8d22).

{{< notice tip >}}
⚡ By the end of this post, you will learn how to:

* **Install** Fastdup and run it on your local machine.
* Find **duplicate** and **anomalies** in your dataset.
* Identify **wrong/confusing labels** in your dataset. 
* Uncover **data leak** in your dataset.

📝 **NOTE**: All codes used in the post are on my [Github repo](https://github.com/dnth/fastdup-manage-clean-curate-blogpost). Alternatively, you can [run this example in Colab](https://github.com/dnth/fastdup-manage-clean-curate-blogpost/blob/main/clean.ipynb).

{{< /notice >}}

If that looks interesting, let's dive in.

### 📖 Installation
To start, run:

```bash
pip install fastdup
```
Feel free to use the latest version available.
I'm running `fastdup==0.189` for this post.

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


### 🏋️‍♀️ Fastdup in Action: Discovering Data Issues
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
+ `report/` -- Directory to save the output generated by Fastdup.

📝 **NOTE**: For simplicity, I've also included the datasets in my [Github repo](https://github.com/dnth/fastdup-manage-clean-curate-blogpost).
{{< /notice >}}

To start checking through the images, create a Jupyter notebook and run:

```python
import fastdup
fastdup.run(input_dir='scene_classification/data/train_set/', 
            work_dir="scene_classification/report/train/")
```

{{< notice note >}}
Parameters for the `run` method:
* `input_dir` -- Path to the folder containing images. In this post, we are checking the training dataset.
* `work_dir` -- **Optional**. Path to save the outputs from the run. If not specified, the output will be saved to the current directory.

**📝 NOTE**: More info on other parameters [here](https://visual-layer.github.io/fastdup/#fastdup.run).
{{< /notice >}}

Fastdup will run through all images in the folder to check for issues.
How long it takes depends on how powerful is your CPU. 
On my machine, with an Intel Core™ i9-11900 it takes **under 1 minute** to check through (approx. 25,000) images in the folder 🤯.

<!-- {{< notice tip >}}
In this post, I'm only running on the `train_set` folder to illustrate what's possible. 

Feel free to repeat the steps for `valid_set` and `test_set` by pointing `input_dir` and `work_dir` to the appropriate folders. 

{{< /notice >}} -->

Once complete, you'll find a bunch of output files in the `work_dir` folder.
We can now visualize them accordingly.

The upcoming sections show how you can visualize [duplicates](#-duplicates), [anomalies](#-anomalies), [confusing labels](#-wrong-or-confusing-labels) 
and [data leakage](#-data-leakage). 
Read on.

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
Parameters for `create_duplicates_gallery` method:
* `similarity_file` -- A `.csv` file with the computer similarity generated by the `run` method.
* `save_path` -- Path to save the visualization. Defaults `'./'`.
* `num_images` -- The max number of images to display. Defaults to `50`. For brevity, I've set it to `5`.

**📝 NOTE**: More info on other parameters [here](https://visual-layer.github.io/fastdup/#fastdup.create_duplicates_gallery).

{{< /notice >}}

You'd see something like the following 👇

{{< include_html "./content/portfolio/fastdup_manage_clean_curate/similarity.html" >}}


{{< notice info >}}

We can already spot a few issues in the `train_set`:

* On `row 1`, note that `19255.jpg` and `7872.jpg`
are **duplicates of the same class**. We know this by the `Distance` value of `1.0`. 
You can also see that they are exactly the same side-by-side. The same with `row 2`.

* On `row 0`, images `9769.jpg` and `7293.jpg` are exact copies but they exist in both the `buildings` and `street` folders!
The same can be seen on `row 3` and `row 4`. 
These are **duplicate images but labeled as different classes** and will end up confusing your model!

{{< /notice >}}

For brevity, I've only shown 5 rows, if you run the code increasing `num_images`, you'd find more!

Duplicate images do not provide value to your model, they take up hard drive space and increase your training time.
Eliminating these images improves your model performance, and reduces cloud billing costs for training and storage.

Plus, you save valuable time (and sleepless nights 🤷‍♂️) to train and troubleshoot your models down the pipeline. 

You can choose to remove the images by hand (e.g. going through them one by one and hitting the delete key on your keyboard.) There are cases you might want to do so. But Fastdup also provides a convenient method to remove them programmatically.

{{< notice warning >}}
The following code will **delete all duplicate images** from your folder. I recommend setting `dry_run=True` to see which files will be deleted.

📝 **NOTE**: Checkout the [Fastdup documentation](https://visual-layer.github.io/fastdup/#fastdup.delete_components) to learn more about the parameters you can tweak.
{{< /notice >}}

```python
top_components = fastdup.find_top_components(work_dir="scene_classification_clean/report/")
fastdup.delete_components(top_components, dry_run=False)
```

In Fastdup, a **component** is a **cluster** of similar images.
The snippet above removes duplicates of the same images (from the top cluster) ensuring you only have one copy of the image in your dataset.


That's how easy it is to find duplicate images and remove them from your dataset! 
Let's see if we can find more issues.

#### 🦄 Anomalies
Similar to duplicates, it's easy to visualize anomalies in your dataset:

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

📝 **NOTE**: Run the code snippet and increase the `num_images` parameter to see more anomalies. 
Also, repeat this with `valid_set` and see if there are more.
{{< /notice >}}

All the other images above don't look too convincing to me either.
I guess you can evaluate the rest if they belong to the right classes as labeled. 
Now let's see how we can programmatically remove them.

{{< notice warning >}}
The following code will **delete all outliers** from your folder. I recommend setting `dry_run=True` to see which files will be deleted.

📝 **NOTE**: Checkout the [Fastdup documentation](https://visual-layer.github.io/fastdup/#fastdup.delete_or_retag_stats_outliers) to learn more about the function parameters.
{{< /notice >}}

```python
fastdup.delete_or_retag_stats_outliers(stats_file="scene_classification_clean/report/outliers.csv", 
                                       metric='distance', filename_col='from', 
                                       lower_threshold=0.6, dry_run=False)
```
The above command removes all images with the `distance` value of `0.6` or below.

What value you pick for the `lower_threshold` will depend on the dataset. In this example, I notice that as `distance` go higher than `0.6`, the images look less like outliers. 

This isn't a foolproof solution, but it should remove the bulk of anomalies present in your dataset.

#### 💆 Wrong or Confusing Labels
One of my favorite capabilities of Fastdup is finding wrong or confusing labels.
Similar to previous sections, we can simply run:

```python
df = fastdup.create_similarity_gallery(similarity_file="scene_classification/report/train/similarity.csv", 
                                  save_path="scene_classification/report/train/", 
                                  get_label_func=lambda x: x.split('/')[-2], 
                                  num_images=5, max_width=180, slice='label_score', 
                                  descending=False)
HTML('./scene_classification/report/train/topk_similarity.html')

```

{{< notice note >}}
In case the dataset is labeled, you can specify the label using the function `get_label_func`. 

📝 **NOTE**: Check out the [Fastdup documentation](https://visual-layer.github.io/fastdup/#fastdup.create_similarity_gallery) for parameters description.
{{< /notice >}}

You'd see something like 👇

{{< include_html "./content/portfolio/fastdup_manage_clean_curate/topk_similarity.html" >}}

Under the hood, Fastdup finds images that are similar to one another at the embedding level but are assigned different labels.

A `score` metric is computed to reflect how similar the query image to the most similar images in terms of class label.

A **high** `score` means the query image looks similar to other images in the same class. Conversely, a **low** `score` indicates the query image is similar
to images from other classes.

{{< notice info >}}
What can we see in the table above?

* On the top row, we find that `3279.jpg` is labeled `forest` but looks very similar to `mountains`. 
* On the remaining rows, we see confusing labels between `glacier` and `mountains`.

{{< /notice >}}

It is important to address these confusing labels because if the training data contains confusing or incorrect labels, it can negatively impact the performance of the model.

{{< notice tip >}}
You can repeat the steps to find duplicates, anomalies, and problematic labels for the `valid_set` and `test_set`. 
To do so, you'd have to call the `run` method again specifying the appropriate dataset folders.
{{< /notice >}}

Using Fastdup we can delete or retag these images.

{{< notice warning >}}
The following code will **delete all images with wrong labels** from your folder. I recommend setting `dry_run=True` to see which files will be deleted.

📝 **NOTE**: Checkout the [Fastdup documentation](https://visual-layer.github.io/fastdup/#fastdup.delete_or_retag_stats_outliers) to learn more about the parameters.
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

📝 **NOTE**: Checkout the [Fastdup documentation](https://visual-layer.github.io/fastdup/#fastdup.delete_or_retag_stats_outliers) to learn more about the parameters.

{{< /notice >}}

#### 🚰 Data Leakage
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

This is bad news. We just uncovered a **train-validation data leakage**! 

This is a common reason a model performs all too well during training and fails in production because the model might just 
memorize the training set without generalizing to unseen data. It's important to make sure the training and validation sets do not contain duplicates!

<!-- {{< notice tip >}}
* A validation set consists of **representative** and **non-overlapping** samples from the train set and is used to evaluate models during training.
* Overlapping images in the train and validation set may lead to poor performance on new data.
* The way we craft our validation set is extremely important to ensure the model does not overfit. 
{{< /notice >}} -->

Spending time crafting your validation set takes a little effort, but will pay off well in the future.
Rachel Thomas from [Fastai](https://www.fast.ai/) wrote a good piece on [how to craft
a good validation set](https://www.fast.ai/posts/2017-11-13-validation-sets.html).

You can remove the duplicate images using the `delete_components` method as shown in the [Duplicates section](#-duplicates).

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
{{< /notice >}}

By using Fastdup and cleaning your dataset, you saved yourself:
* Unnecessary labeling cost.
* Long computation/training time.
* Headaches from debugging model predictions due to problems in data.

I believe Fastdup is one of the easiest tools to get started for data cleaning. 
It’s a low-hanging fruit and ought to be in your toolkit if you’re working with image datasets.

If you're interested to learn more, I've trained a deep learning classification model on the clean version of the data using [Fastai](https://www.fast.ai/). View the training notebook [here](https://github.com/dnth/fastdup-manage-clean-curate-blogpost/blob/main/train_clean.ipynb). 
The accuracy on the validation set is approximately **94.9%** - comparable to the [winning solutions of the competition](https://medium.com/@afzalsayed96/1st-place-solution-for-intel-scene-classification-challenge-c95cf941f8ed) (96.48% with ensembling).

I hope you've enjoyed and learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter/LinkedIn post or [drop me a message](https://dicksonneoh.com/contact/).

{{< tweet dicksonneoh7 1618622445581393920>}}

<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:activity:7024313080490721280/" height="1760" width="550" frameborder="0" allowfullscreen="" title="Embedded post"></iframe>

