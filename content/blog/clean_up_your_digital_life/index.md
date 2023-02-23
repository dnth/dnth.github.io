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

{{< notice tip >}}
Fastdup offers an enterprise edition of the tool that lets you do more. [Learn more here](https://www.visual-layer.com/).
{{< /notice >}}

It's easy to get started and use. 
The [authors of Fastdup](https://www.visual-layer.com/) even used it to uncover over **1.2M duplicates** and **104K data train/validation leaks** in the ImageNet-21K dataset [here](https://medium.com/@amiralush/large-image-datasets-today-are-a-mess-e3ea4c9e8d22).


If all that looks interesting, let's get started with..

### ☕ Messy Images

![img](https://media.giphy.com/media/10zsjaH4g0GgmY/giphy.gif)

As we are going to clean up messy albums, the first step is to download the photos from your Google Photos, Onedrive or wherever you have your images into your local drive.

I don't have a massive photo collection, so I’ll be using a dataset from [Kaggle](https://www.kaggle.com/datasets/duttadebadri/image-classification) that was scraped off Google Download them into a folder on your computer. The contributor [Debadri Dutta](https://www.linkedin.com/in/debadridtt/) has a knack of photography and travelling. 

Here are a few sample images.

{{< figure_resizing src="sample.png" caption="Sample images scraped from Google." link="https://www.kaggle.com/datasets/duttadebadri/image-classification" >}}


With the images downloaded locally let's organize them in a folder.
Here's how the folders look on my computer.

```tree
├── images
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

### ❌ Duplicate Photos

### 🗂 Clustering Similar Shots

### 🤳 Dark/Bright Blurry Shots

### 🔓 Conclusion


{{< notice tip >}}
In this blog post, I’ve shown you how to use Fastdup to programmatically -

- Identify duplicate or near identical photos.
- Cluster similar shots together.
- Filter out unnecessary photos that take up storage space.
{{< /notice >}}