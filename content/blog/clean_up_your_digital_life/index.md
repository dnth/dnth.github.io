---
title: "Clean Up Your Digital Life: Simplify Your Photo Organization and Say Goodbye to Photo Clutter"
date: 2023-02-23T11:00:15+08:00
featureImage: images/blog/clean_up_your_digital_life/thumbnail.gif
postImage: images/blog/clean_up_your_digital_life/post_image.gif
tags: ["fastdup", "google-images", "Python"]
categories: ["data-cleaning", "clustering", "duplicate-detection", "blur-detection", "photo-organization"]
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
But over time, these photos will just clutter your devices taking up valuable disk space and [slowing down your device](https://askleo.com/deal-with-computer-clutter-to-speed-things-up/).

Also, think about these -
+ It's difficult to find specific photos when your collection is in a mess.
+ Organizing your collection saves you time spending searching for photos.
+ An organized photo collection can be a source of pride especially when you share them.
+ Digital clutter not only affects your device but also [impacts you psychologically](https://www.bbc.com/future/article/20190104-are-you-a-digital-hoarder).

So consider cleaning up your digital clutter, because it pays in the long run.

If you're convinced, now comes the next hurdle.

{{% blockquote %}}
Spending hours sorting through your photos and cleaning them is a pain. 
Nobody has time for that. We’re busy people.
{{% /blockquote %}}

Don't fret, that's what this post is about.
In this post, I'll show you how to tidy up your digital life by organizing your photo collection and not spending an entire weekend doing it.

{{< notice tip >}}
💫 Here's what you'll learn by the end -

- How to isolate **corrupted** images in your photo album.
- How to identify **duplicates** in your photo album.
- How to filter out photos that are too **dark**, too **bright**, or **blurry**.
- How to **cluster** similar-looking shots together.
- How to **bulk-delete** photos.

📝 **NOTE**: All codes used in the post are on my [Github repository](https://github.com/dnth/clean-up-digital-life-fastdup-blogpost).

{{< /notice >}}

### ⚡ fastdup

[fastdup](https://github.com/visual-layer/fastdup) is a tool that let you gain insights from a large image/video collection. 

You can manage, clean, and curate your images at scale on your local machine event with a single CPU.
fastdup lets you clean visual data with ease, freeing up valuable resources and time. 

Here are some superpowers you get with fastdup - it lets you identify:

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

I'm running `fastdup==0.903` and `Python 3.10` for this post.
Feel free to use the latest version available.

After the installation completes, you can now import fastdup in your Python console and start the run.

```python
import fastdup
work_dir = "./fastdup_report"
images_dir = "./images"

fd = fastdup.create(work_dir, images_dir)
fd.run()

```

{{< notice note >}}
* `images_dir` -- Path to the folder containing images.
* `work_dir` -- Path to save the outputs from the run.

**📝 NOTE**: More info on other parameters on the [docs page](https://visual-layer.readme.io/docs/v1-api#fastdup.engine.Fastdup.run). 
{{< /notice >}}

This starts the process of detecting all issues on the images in `images_dir`.
Depending on your CPU power, this may take a few seconds to a few minutes to complete.

On my machine, with an Intel Core™ i9-11900 it takes **under 1 minute** to check through (approx. 35,000) images in the folder 🤯.

Once the run completes, you'll find the `work_dir` populated with all files from the run.

{{< notice tip >}}
fastdup recommends running the commands in a Python console and **NOT** in a Jupyter notebook.
{{< /notice >}}

Personally, I find no issues running the commands in a notebook. But beware that the notebook size can be large especially if there are lots of images rendered.

Once the run is complete, we can visualize the issues. 

For a summary, run

```python
fd.summary()
```

Here are some useful information from the summary.
 
 + Dataset contains 35136 images.
 + Valid images are 99.83% (35,077) of the data, invalid are 0.17% (59) of the data'
 + 2.15% (756) belong to 12 similarity clusters (components).
 + Largest cluster has 16 (0.05%) images.
 + 6.16% (2,163) of images are possible outliers, and fall in the bottom 5.00% of similarity values.

There are a few issues we can already spot there but let's start with 👇


### 🚫 Invalid Images

Invalid images are files that cannot be read by fastdup. Chances are, they are corrupted images.

We have 59 of them according to the summary.
To get the list of invalid images, run:

```python
fd.invalid_instances()
```

which outputs

{{< figure_resizing src="invalid.png" link="invalid.png" >}}

I tried to open this images on my machine, but they could not be viewed.

{{< figure_resizing src="sample_invalid.png" link="sample_invalid.png" >}}

Invalid images can't be used but takes up disk space. 
There's only one way to deal with it - Delete.


To delete corrupted images with fastdup, let's collect the images into a list:

```python
invalid_images = fd.invalid_instances()
list_of_invalid_images = invalid_images['img_filename'].to_list()
```

`images_to_delete` now contains a list of file directories to be deleted.

```python
['art and culture/145.jpg',
 'art and culture/148 (9).jpg',
 'art and culture/155 (3).jpg',
 'art and culture/156 (5).jpg',
 ...
 ...
 ...
 'art and culture/98 (5).jpg',
 'food and d rinks/1.jpg',
 'food and d rinks/28 (2).jpg',
 'food and d rinks/325 (3).jpg',
 'food and d rinks/424.jpg']
```

What's left to do next is to write a function to delete images in `list_of_invalid_images`.

{{< notice warning >}}
The following code will **DELETE ALL** corrupted images specified in `list_of_invalid_images`.
I recommend **making a backup** of your existing dataset before proceeding.
{{< /notice >}}

```python
from pathlib import Path

def delete_images(file_paths):
    for file_path in file_paths:
        path = images_dir / Path(file_path)
        if path.is_file():
            print(f"Deleting {path}")
            path.unlink()
```

And call the function:
```python
delete_images(list_of_invalid_images)
```

Just like that, we've deleted all corrupted images from our dataset!

{{< notice tip >}}
You can optionally choose move the images to another folder instead of deleting like what we did above.
{{< /notice >}}

We can do that with the following function:
```python
import shutil
from pathlib import Path

def move_images_to_folder(file_paths, folder_name="invalid_images"):
    corrupted_images_dir = Path(folder_name)
    corrupted_images_dir.mkdir(exist_ok=True)  # create the directory if it doesn't exist

    for file_path in file_paths:
        path = images_dir / Path(file_path)
        if path.is_file():
            new_path = corrupted_images_dir / Path(file_path)
            new_path.parent.mkdir(parents=True, exist_ok=True)  # create the parent directory if it doesn't exist
            print(f"Moving {path} to {new_path}")
            shutil.move(str(path), str(new_path))
```

And call the function:

```python
move_images_to_folder(list_of_invalid_images)
```

This should move the invalid images in to the `folder_name` directory.

### 👯‍♂️ Duplicate Images
To view the duplicate photos run:

```python
fd.vis.duplicates_gallery()
```

If you're running this in a Jupyter notebook, you'll see something like the following.
{{< figure_resizing src="duplicates.png" link="duplicates.png" caption="A generated gallery of duplicates." >}}

{{< notice note >}}
You can optionally specify `num_images` -- The max number of images to display. Defaults to `20`.


**📝 NOTE**: More info on other parameters [here](https://visual-layer.readme.io/docs/v1-api#fastdup.fastdup_galleries.FastdupVisualizer.duplicates_gallery).

{{< /notice >}}


In the visualization above we see that there are exact copies residing in different folders within the `images_dir`.

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
fd.vis.stats_gallery(metric='dark')
```

The above snippet sorts all the photos in your folder following ascending `mean` values. So the darker images (lower `mean` value) should appear at the top.

{{< figure_resizing src="dark.png" link="dark.png" caption="A generated gallery of dark images." >}}

The first 3 images (totally black) are classic. I always find these somewhere in my albums due to accidental press when the phone is in my pocket.

I leave it to you to judge if you'd keep or discard the rest of the images.


Conversely, get the brightest images on top with:

```python
fd.vis.stats_gallery(metric='bright')
```

{{< figure_resizing src="bright.png" link="bright.png" caption="A generated gallery of bright images." >}}

Again, see the first 3 images (totally white) which happens sometimes when your shots are overexposed.


And next let's sort our album with the `blur` metric.
You've guessed it, this sorts our album with the most blurry image on top.


```python
fd.vis.stats_gallery(metric='blur')
```

{{< figure_resizing src="blur.png" link="blur.png" caption="A generated gallery of blur images." >}}


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
fd.vis.component_gallery()
```

And you'll find something like the following.

{{< figure_resizing src="components.png" caption="" link="components.png" >}}

Above, I've shown you three examples of similar-looking shots grouped together with the file path of each image.
It's up to you to decide what to do with the similar looking shots. Not going to use them? Delete. Otherwise you can also keep them organized in a folder of some sort.

{{< notice tip >}}
Check out the full output of the above code in the [notebook](https://github.com/dnth/clean-up-digital-life-fastdup-blogpost/blob/main/fastdup_analyze.ipynb).
{{< /notice >}}

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