---
title: "Clean Up Your Digital Life: Simplify Your Photo Organization and Say Goodbye to Photo Clutter"
date: 2023-02-23T11:00:15+08:00
featureImage: images/blog/clean_up_your_digital_life/thumbnail.gif
postImage: images/blog/clean_up_your_digital_life/post_image.gif
tags: ["fastdup", "google-images", "Python"]
categories: ["data-cleaning", "clustering", "duplicate-detection", "blur-detection"]
toc: true
socialshare: true
description: "Even pros have dark, blurry & duplicate shots. But disorganization can make it hard to find those special memories. Let's fix that."
images : 
- images/blog/clean_up_your_digital_life/post_image.gif
---

### ✅ Motivation

In today's world of selfies and Instagram, we all take tons of photos on our phones, cameras, and other gadgets. 

But let's be real, it's easy for our photo collections to become a chaotic mess, making it impossible to find that one special memory. 

I mean, I've got *gigabytes* of photos on my Google Photo app filled with dark shots, overly exposed shots, blurry shots, and tons of duplicate stills.

And let's face it, what we post on Instagram vs what's *behind the scenes* can be wildly different.

{{< figure_resizing src="meme.png" caption="Me vs my photo album." >}}

I know, you'll say that there's no harm in keeping those extra selfies in your phone. Right?

Not in the short term.
But over time, these photos will just clutter your devices taking up valuable disk space and [slow down your device](https://askleo.com/deal-with-computer-clutter-to-speed-things-up/).

Also, consider these -
+ It's difficult to find specific photos when your collection is in a mess.
+ Organizing your collection saves you time spending searching for photos.
+ An organized photo collection can be a source of pride especially when you share them.
+ Digital clutter not only affects your device but also [impacts you psychologically](https://www.bbc.com/future/article/20190104-are-you-a-digital-hoarder).

So consider cleaning up your digital clutter, because it pays in the long run.

If you're convinced, then comes the next hurdle.

{{% blockquote %}}
Spending hours sorting through your photos and cleaning them is a pain. 
Nobody has time for that. We’re busy people.
{{% /blockquote %}}

Don't fret, that's what this post is about.
In this post, I'll show you how to tidy up your digital life by organizing your photo collection and not spending an entire weekend doing it.

{{< notice tip >}}
💫 Here's what you'll learn by the end -

- How to identify duplicates in your photo album using Python code.
- How to filter out photos that are too dark, too bright, or blurry.
- How to group similar-looking shots together.
- How to bulk-delete duplicate photos.

📝 **NOTE**: All codes used in the post are on my [Github repository](https://github.com/dnth/clean-up-digital-life-fastdup-blogpost).

{{< /notice >}}

### ⚡ fastdup

[fastdup](https://github.com/visual-layer/fastdup) is a tool that let us gain insights from a large image/video collection. 

You can manage, clean, and curate your images at scale on your local machine event with a single CPU.
fastdup lets you clean visual data with ease, freeing up valuable resources and time. 

Here are some superpowers you get with fastdup.
It lets you identify:

{{< figure_resizing src="features.png" caption="fastdup superpowers. Source: fastdup GitHub." link="https://github.com/visual-layer/fastdup" >}}

In short, fastdup is 👇
* **Unsupervised**: fits any visual dataset.
* **Scalable**: handles 400M images on a single machine.
* **Efficient**: works on CPU (even on Google Colab with only 2 CPU cores!).
* **Low Cost**: can process 12M images on a $1 cloud machine budget.

🌟 The best part? fastdup is **free**.

{{< notice info >}}
fastdup also offers an Enterprise edition of the tool that lets you do more. Find out [here](https://www.visual-layer.com/).
{{< /notice >}}

If that looks interesting, let's get started with.. 👇

### ☕ Messy Images

![img](https://media.giphy.com/media/10zsjaH4g0GgmY/giphy.gif)

As we are going to clean up messy albums, the first step is to download the photos from your Google Photos, Onedrive, or whatever cloud service you use into your local drive.

I don't have a massive photo collection, so I’ll be using an image collection from [Kaggle](https://www.kaggle.com/datasets/duttadebadri/image-classification) that was scraped off Google Images.

The contributor [Debadri Dutta](https://www.linkedin.com/in/debadridtt/) has a knack for photography and traveling. A lot of the images from the collection are uploaded by users on social media.
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
+ `fastdup_report/` -- Directory to save the output generated by fastdup.
+ `fastdup_analyze.ipynb` -- Jupyter notebook to run fastdup.

📝 **NOTE**: If you'd like to follow along with the example on this post download the images to your desktop from Kaggle [here](https://www.kaggle.com/datasets/duttadebadri/image-classification) into the `images/` directory.
{{< /notice >}}

With the folders in place let's get working.

### 🧮 Install and Run
First, let's install fastdup with:

```bash
pip install fastdup
```

I'm running `fastdup==0.210` and `Python 3.9` for this post.
Feel free to use the latest version available.

After the installation completes, you can now import fastdup in your Python console and start the run.

```python
import fastdup
fastdup.run(input_dir="./images", 
            work_dir="./fastdup_report",
            turi_param='ccthreshold=0.88')

```

{{< notice note >}}
* `input_dir` -- Path to the folder containing images. In this post, we are checking the training dataset.
* `work_dir` -- **Optional**. Path to save the outputs from the run. If not specified, the output will be saved to the current directory.
* `turi_param` -- **Optional**. Used to control how the images are clustered together. Higher `ccthreshold` results in lesser images grouped together. Default value: `'ccthreshold=0.98'`.

**📝 NOTE**: More info on other parameters on the [docs page](https://visual-layer.github.io/fastdup/#fastdup.run). 
{{< /notice >}}

This starts the process of detecting all issues on the images in `input_dir`.
This process may take anywhere from a few seconds to a few minutes to complete.

Since fastdup runs on CPUs, the time it takes depends a lot on your CPU power.
On my machine, with an Intel Core™ i9-11900 it takes **under 1 minute** to check through (approx. 35,000) images in the folder 🤯.

Once the run completes, you'll find the `work_dir` populated with all files from the run.

{{< notice tip >}}
fastdup recommends running the commands in a Python console and **NOT** in a Jupyter notebook.
{{< /notice >}}

Personally, I find no issues running the commands in a notebook. But beware that the notebook size can be large especially if there are lots of images rendered.

Once the run is complete, we can visualize the issues. 

Let's start with 👇

### ❌ Duplicate Photos
To view the duplicate photos run:

```python
import fastdup
from IPython.display import HTML
fastdup.create_duplicates_gallery(similarity_file='./fastdup_report/similarity.csv',
                                  save_path='./fastdup_report/', 
                                  num_images=20)

HTML('./fastdup_report/similarity.html')
```
{{< notice note >}}
* `similarity_file` -- A `.csv` file with the computer similarity generated by the `run` method.
* `save_path` -- Path to save the visualization. Defaults `'./'`.
* `num_images` -- The max number of images to display. Defaults to `50`.

**📝 NOTE**: More info on other parameters [here](https://visual-layer.github.io/fastdup/#fastdup.create_duplicates_gallery).

{{< /notice >}}

This generates a file `similarity.html` in the `save_path` directory. You can open the `.html` file in your browser to view it.

If you're running this in a Jupyter notebook, you'll see something like the following.
{{< figure_resizing src="duplicates.png" caption="A generated gallery of duplicates." >}}


In <span style="color:red">❶</span>,  <span style="color:red">❷</span> and <span style="color:red">❸</span> we find duplicate images with a `Distance` score of `1.0`. These are **EXACT** copies of one another taking up valuable disk space.

So what do we do about it?
You can either refer to the file name and delete the duplicate images by hand.

Or

Use a convenient function in fastdup to bulk delete images that are **EXACT** copies.

```python
top_components = fastdup.find_top_components(work_dir="fastdup_report/")
fastdup.delete_components(top_components, dry_run=False)
```

{{< notice warning >}}
The above code will **delete all duplicate images** from your folder. 
To not risk losing any data, I recommend making a backup copy of your files before deleting.

Setting `dry_run=True` in the function tells fastdup to list all files that will be deleted. Once you're sure, then set `dry_run=False` to perform the actual deletion.

📝 **NOTE**: Check out the [fastdup documentation](https://visual-layer.github.io/fastdup/#fastdup.delete_components) to learn more about the parameters you can tweak.
{{< /notice >}}

Just like that, we've eliminated duplicates from the album! In this post, I found a total of **1929 fully identical images**!

Now on to the next common problem in photo albums 👇

### 🤳 Dark, Bright, and Blurry Shots

Let's be real, even pros have overly dark bright, and blurry shots in their albums. 
These shots are probably not going to be used and hog your storage space.

With fastdup you can filter them out with:


```python
fastdup.create_stats_gallery('./fastdup_report/atrain_stats.csv', 
                             save_path='./fastdup_report', descending=False,
                             max_width=400, metric='mean')
HTML('./fastdup_report/mean.html')
```

The above snippet sorts all the photos in your folder following ascending `mean` values. So the darker images (lower `mean` value) should appear at the top.

{{< figure_resizing src="dark.png" caption="A generated gallery of dark images." >}}

Image <span style="color:red">❶</span> (totally black) is classic. I always find these somewhere in my albums due to accidental press when the phone is in my pocket.

Image <span style="color:red">❷</span> and <span style="color:red">❸</span> looks legit to me, but I leave it to you to judge if you'd keep or discard those.


If we change the parameter to `descending=True` in the above snippet, it should show the opposite ie. sorting from the brightest image first.

```python
fastdup.create_stats_gallery('./fastdup_report/atrain_stats.csv', 
                             save_path='./fastdup_report', 
                             descending=True,
                             max_width=400, metric='mean')
HTML('./fastdup_report/mean.html')
```

{{< figure_resizing src="bright.png" caption="A generated gallery of bright images." >}}

Again we see Image <span style="color:red">❶</span> (totally white) which happens sometimes when your shots are overexposed.

Image <span style="color:red">❷</span> and <span style="color:red">❸</span> looks like it's a random text document scraped off the internet. Probably irrelevant. I'd remove those.

And next let's sort our album with the `blur` metric.
You've guessed it, this sorts our album with the most blurry image on top.


```python
fastdup.create_stats_gallery('./fastdup_report/atrain_stats.csv', 
                             save_path='./fastdup_report', 
                             descending=False,
                             max_width=400, metric='blur')
HTML('./fastdup_report/blur.html')
```

{{< figure_resizing src="blur.png" caption="A generated gallery of blur images." >}}


There are more ways we can view our photos using statistical metrics. So you can change the `metric` argument to:

+ `blur` -- Sort by blurriness. 
+ `mean` -- Sort by mean value.
+ `min` -- Sort by minimum value.
+ `max` -- Sort by maximum value.
+ `stdv` -- Sort by standard deviation value.

View other examples [here](https://github.com/visual-layer/fastdup/blob/main/examples/fastdup_image_stats.ipynb).

{{< notice tip >}}
Try running with `metric='stdv'`. You'll find images that lie outside of the standard deviation and potentially find anomalies in them.
{{< /notice >}}


### 🗂 Clustering Similar Shots
This is one of my favorite functions in fastdup.

With all the thousands of photos in one album, it will be interesting to group similar shots to assess them as a whole.

It's also easier to identify patterns and trends in these similar shots. Or you may find that these shots are just redundant shots that will not be used.

To group similar shots together run:

```python
fastdup.create_components_gallery(work_dir='./fastdup_report/',
                                  save_path='./fastdup_report/')
HTML('./fastdup_report/components.html')
```

And you'll find something like the following.

{{< figure_resizing src="cluster_160.png" caption="" >}}
{{< figure_resizing src="cluster_6667.png" caption="" >}}
{{< figure_resizing src="cluster_16356.png" caption="A gallery of similar shots." >}}

Above, I've shown you three examples of similar-looking shots grouped. There are more. Check them out in the [notebook](https://github.com/dnth/clean-up-digital-life-fastdup-blogpost/blob/main/fastdup_analyze.ipynb).

### 🔓 Conclusion

<div style="width:480px"><iframe allow="fullscreen" frameBorder="0" height="270" src="https://giphy.com/embed/8qxVaAQ9jycFRhWITO/video" width="480"></iframe></div>

Cleaning up your digital photo collection is an important step towards simplifying your digital life. 

Disorganized photos can take up valuable storage space, slow down your device's performance, and make it difficult to find specific photos when you need them. 

In this blog post, I’ve shown you how to use fastdup to programmatically clean your photo collections without spending a lot of time.

{{< notice tip >}}
💫 Here's what we learned -

- How to identify duplicates in your photo album using Python code.
- How to filter out photos that are too dark, too bright, or blurry.
- How to group similar-looking shots together.
- How to bulk-delete photos.

📝 **NOTE**: All codes used in the post are on my [Github repository](https://github.com/dnth/clean-up-digital-life-fastdup-blogpost).

{{< /notice >}}

By using fastdup to identify and delete duplicate and unwanted photos, and clustering similar photos for easy organization, you can save time and energy and enjoy a well-organized digital photo collection.

I hope you've enjoyed and learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter/LinkedIn post or [drop me a message](https://dicksonneoh.com/contact/).

{{< tweet dicksonneoh7 1618622445581393920>}}

<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:activity:7024313080490721280/" height="1800" width="504" onload='javascript:(function(o){o.style.height=o.contentWindow.document.body.scrollHeight+"px";}(this));' frameborder="0" allowfullscreen="" title="Embedded post"></iframe>