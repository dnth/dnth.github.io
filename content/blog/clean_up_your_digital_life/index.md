---
title: "Clean Up Your Digital Life: Simplify Your Photo Organization and Say Goodbye to Photo Clutter"
date: 2023-02-23T11:00:15+08:00
featureImage: images/blog/clean_up_your_digital_life/thumbnail.gif
postImage: images/blog/clean_up_your_digital_life/post_image.gif
tags: ["Fastdup", "google-images", "Python", "C"]
categories: ["data-cleaning", "clustering"]
toc: true
socialshare: true
description: "Even pros have dark, blurry & duplicate shots. But disorganization can make it hard to find those special memories. Let's fix that."
images : 
- images/blog/clean_up_your_digital_life/post_image.gif
---

### ✅ Motivation

In today's world of selfies and Instagram, we all take tons of photos on our phones, cameras, and other gadgets. But let's be real, it's easy for our photo collections to become a chaotic mess, making it impossible to find that one special memory. 

I’ve had gigabytes in my Google Photo filled with dark shots taken accidentally, overly exposed shots, blurry shots, and tons of duplicate shots. Let’s be honest, we all tap on the camera button multiple times to take burst shots in an attempt to capture the best look.

{{< figure_resizing src="meme.png" caption="Me and my album." >}}


Why photo organization is important

- Disorganized photos make it difficult to find specific photos when you need them.
- Photo organization saves time and energy when searching for specific photos.
- Digital clutter can take up valuable storage space on your devices, which can slow down their performance.
- A well-organized digital photo collection can be a source of pride and enjoyment.
- It's easier to share organized photos with others through social media or physical photo albums or prints.

Sorting through your photos and deleting unwanted photos can be a time-consuming task and nobody wants to spend hours doing just that. We’re busy people.

Don't fret, though! With a few easy-peasy steps, you can declutter your photo library and say goodbye to the headache of searching for your favorite pics. In this article, we'll show you how to tidy up your digital life by organizing your photo collection.

This blog will demonstrate how to use Fastup to effectively clean up your photo collection through programming. 

{{< notice tip >}}
We will cover the following topics:

- Identifying duplicate or nearly identical photos.
- Grouping similar photos together and selectively deleting them.
- Filtering out photos that are too dark, too bright, or blurry.


📝 **NOTE**: All codes used in the post are on my [Github repo](https://github.com/dnth/clean-up-digital-life-fastdup-blogpost). Alternatively, you can [run this example in Colab](https://github.com/dnth/fastdup-manage-clean-curate-blogpost/blob/main/clean.ipynb).

{{< /notice >}}

If you have a messy photo collection, you can manually go through the images and remove them one by one. I've done that myself.

But if you have thousands of them.. It will take forever. 
And why do it manually when you can get a machine to do it for you?

Enter 👇

### ⚡ Fastdup

Fastdup is a tool that let us gain insights from a large image/video collection. 

You can manage, clean, and curate your images at scale on your local machine with a single CPU.
It's incredibly easy to use and highly efficient. 
Fastdup lets you clean visual data with ease, freeing up valuable resources and time. 

Here are some superpowers you get with Fastdup.
It lets you identify:

{{< figure_resizing src="features.png" caption="Fastdup superpowers. Source: Fastdup GitHub." link="https://github.com/visual-layer/fastdup" >}}


Fastdup offers a range of benefits that make it the ultimate tool for managing your visual data. 

Here are a few:

+ Efficient: Fastdup's advanced algorithms enable you to quickly identify and eliminate duplicate images and videos, freeing up valuable storage space and ensuring that your data is clean and organized.

+ Easy to use: Fastdup is incredibly user-friendly, with a simple and intuitive interface that makes it easy to manage even large image and video collections.

+ Fast and efficient: With Fastdup, you can gain insights from your visual data quickly and efficiently, saving you valuable time and resources.

+ Scalable: Fastdup is designed to work with large image and video collections, allowing you to manage and curate your data at scale.

The best part? Fastdup is **free**.

{{< notice info >}}
Fastdup offers an enterprise edition of the tool that lets you do more. [Learn more here](https://www.visual-layer.com/).
{{< /notice >}}

It's easy to get started and use. 
The [authors of Fastdup](https://www.visual-layer.com/) even used it to uncover over **1.2M duplicates** and **104K data train/validation leaks** in the ImageNet-21K dataset [here](https://medium.com/@amiralush/large-image-datasets-today-are-a-mess-e3ea4c9e8d22).


If all that looks interesting, let's get started with..

### ☕ Messy Images

![img](https://media.giphy.com/media/10zsjaH4g0GgmY/giphy.gif)

As we are going to clean up messy albums, the first step is to download the photos from your Google Photos, Onedrive or wherever you have your images into your local drive.

I don't have a massive photo collection, so I’ll be using a dataset from [Kaggle](https://www.kaggle.com/datasets/duttadebadri/image-classification) that was scraped off Google Images.

The contributor [Debadri Dutta](https://www.linkedin.com/in/debadridtt/) has a knack of photography and travelling. A lot of the images from the dataset are uploaded by users on social media.
So I thought it would be a good fit to use it for this post.

Here are a few sample images.
{{< figure_resizing src="sample.png" caption="Sample images scraped from Google." link="https://www.kaggle.com/datasets/duttadebadri/image-classification" >}}


With the images downloaded locally let's organize them in a folder.
Here's how the folders look on my computer.

```tree
├── images
|   ├── image001.jpg
|   ├── image002.jpg
|   └── ...
├── fastdup_report
└── fastdup_analyze.ipynb
```

{{< notice note >}}
Description -
+ `images/` -- Folder to store the images.
+ `fastdup_report/` -- Directory to save the output generated by Fastdup.
+ `fastdup_analyze.ipynb` -- Jupyter notebook to run Fastdup.

📝 **NOTE**: If you'd like to follow along with the example on this post download the images into your desktop from Kaggle [here](https://www.kaggle.com/datasets/duttadebadri/image-classification) into the `images/` directory.
{{< /notice >}}

With the folders in place let's get working.

### 🧮 Install and Run
To install, run:

```bash
pip install fastdup
```
Feel free to use the latest version available.
I'm running `fastdup==0.210` for this post.

After the installation completes, you can now import Fastdup in your Python console.

```python
import fastdup
fastdup.run(input_dir="./images", 
            work_dir="./fastdup_report",
            turi_param='ccthreshold=0.88')

```

{{< notice note >}}
Description - 

* `input_dir` -- Path to the folder containing images. In this post, we are checking the training dataset.
* `work_dir` -- **Optional**. Path to save the outputs from the run. If not specified, the output will be saved to the current directory.
* `turi_param` -- **Optional**. Used to control how the images are clustered together. Higher `ccthreshold` results in lesser images grouped together. Default value: `'ccthreshold=0.98'`.

**📝 NOTE**: More info on other parameters on the [docs page](https://visual-layer.github.io/fastdup/#fastdup.run). 

Alternatively run the following in your Python console.
```python
help(fastdup)
``` 


{{< /notice >}}


{{< notice tip >}}
Fastdup recommends running the commands in a Python console and **NOT** in a Jupyter notebook.
{{< /notice >}}

Personally, I find no issues running the commands in a notebook. But beware that the notebook file size can be large especially if there are lots of images rendered.

Fastdup runs on CPUs.
depending on your CPU power, it can take a few seconds to minutes for the run to complete.

On my machine, with an Intel Core™ i9-11900 it takes **under 1 minute** to check through (approx. 35,000) images in the folder 🤯.

Once the run completes, you'll find the `work_dir` populated with all files from the run.

### ❌ Duplicate Photos
Now let's look at the duplicate photos by running the following:

```python
import fastdup
from IPython.display import HTML
fastdup.create_duplicates_gallery(similarity_file='./fastdup_report/similarity.csv',
                                  save_path='./fastdup_report/', 
                                  num_images=20)

HTML('./fastdup_report/similarity.html')
```
{{< notice note >}}
Parameters for `create_duplicates_gallery` method:
* `similarity_file` -- A `.csv` file with the computer similarity generated by the `run` method.
* `save_path` -- Path to save the visualization. Defaults `'./'`.
* `num_images` -- The max number of images to display. Defaults to `50`.

**📝 NOTE**: More info on other parameters [here](https://visual-layer.github.io/fastdup/#fastdup.create_duplicates_gallery).

{{< /notice >}}

This generates a file `similarity.html` in the `save_path` directory. You can open the html file in your browser to view it.

Alternatively, if you're running this in a Jupyter notebook, you'll see something like the following.
{{< figure_resizing src="duplicates.png" caption="A generated gallery of duplicates." >}}


In <span style="color:red">❶</span>,  <span style="color:red">❷</span> and <span style="color:red">❸</span> we find duplicate images with a `Distance` score of `1.0`.

So what do we do about it?
You can either refer to the file name and delete the duplicate images by hand.

Or

Use a convenient function in fastdup to bulk delete images that are EXACT copies.

```python
top_components = fastdup.find_top_components(work_dir="fastdup_report/")
fastdup.delete_components(top_components, dry_run=False)
```

{{< notice warning >}}
The above code will **delete all duplicate images** from your folder. 
I recommend setting `dry_run=True` to see which files will be deleted first and then doing the actual bulk deletion.

📝 **NOTE**: Check out the [Fastdup documentation](https://visual-layer.github.io/fastdup/#fastdup.delete_components) to learn more about the parameters you can tweak.
{{< /notice >}}


### 🤳 Dark/Bright Blurry Shots

{{< figure_resizing src="dark.png" caption="A generated gallery of dark images." >}}

{{< figure_resizing src="bright.png" caption="A generated gallery of bright images." >}}

{{< figure_resizing src="blur.png" caption="A generated gallery of blur images." >}}

### 🗂 Clustering Similar Shots

{{< figure_resizing src="cluster_160.png" caption="" >}}
{{< figure_resizing src="cluster_6667.png" caption="" >}}
{{< figure_resizing src="cluster_16356.png" caption="A gallery of similar shots." >}}

### 🔓 Conclusion
In this blog post, I’ve shown you how to use Fastdup to programmatically clean your photo collections.

{{< notice tip >}}
Specifically, we've seen how to -
- Identify duplicate.
- Identify  dark, bright and blurry shots.
- Cluster near identical photos together.
{{< /notice >}}