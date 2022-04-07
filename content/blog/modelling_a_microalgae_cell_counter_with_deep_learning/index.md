---
title: "Modelling a Microalgae Cell Counter with Deep Learning"
date: 2022-04-06T15:07:15+08:00
featureImage: images/blog/modelling_a_microalgae_cell_counter_with_deep_learning/thumbnail.gif
postImage: images/blog/modelling_a_microalgae_cell_counter_with_deep_learning/post_image.jpg
tags: ["IceVision", "Fast.ai", "counting", "cell"]
categories: ["modelling", "object-detection", "biology"]
toc: true
socialshare: true
---

### 🕶️ Motivation
Numerous biology and medical procedure involve counting cells from images taken with microscope.
Counting cells reveals the concentration of bacteria and viruses and gives vital information on the progress of a disease.
To accomplish the counting, a device known as the hemocytometer or a counting chamber is used.
The hemocytometer creates volumetric grid to divide each region on the image for accurate counting.


The following YouTube video illustrates the counting process using a hemocytometer.
{{< youtube WWS9sZbGj6A >}}


As shown in the video, each cell in the image has to be manually and meticulously counted.
This process may be slow and is prone to human error.
What if we could automate the counting by using an intelligent deep learning model?

In this blog post, we will see how easy it gets for anyone to use the IceVision computer vision library and quickly train a sophisticated deep learning model to count microalgae cells.

For the purpose of this post, I've acquired image samples from a lab with a colony of microalgae cells. 
The following image shows a sample image of the cells as seen from a microscope.
The microalgae cells are in green.
{{< figure_resizing src="hemocytometer.jpg" >}}

By the end of this blog post, you should be able to train your own microalgae cell (or any other objects really) counter with the steps that I will walk you through.
Did I mention that all the tools used in this project are open-source and free of charge? Yes!! 
If you're ready let's begin.


### ⚙️ Installation
We will be using a computer vision library known as IceVision - a computer vision focused library built to work with Fastai. 
I highly recommend that you use a virtual environment like Anaconda to install the package. 
[Here](https://www.geeksforgeeks.org/set-up-virtual-environment-for-python-using-anaconda/) is how to set it up.

Once setup, we are ready to install all the packages for this blog post.
First clone the Git repository:

```bash
git clone https://github.com/dnth/microalgae-cell-counter-blogpost
```

Next, navigate into the directory:

```bash
cd microalgae-cell-counter-blogpost/
```

Install IceVision:

```bash
bash icevision_install.sh cuda11 0.12.0
```

Depending on your system `CUDA` version, you may want to change `cuda11` to `cuda10` on older systems. 
The number following the cuda version is the version of IceVision we are installing. 
The version I'm using for this blog post is `0.12.0`.
You can alternatively specify `master` to install the bleeding edge version of IceVision from the master branch on Github.

The installation may take a couple of minutes depending on the speed of your internet connection.
Allow the installation to complete before proceeding to the next step.



### 🔖 Preparing the dataset
Before embarking on any machine learning work, we must ensure that we have a dataset to work on.
Our task at hand is to construct a model that can count microalgaes. 
Since there are no public dataset available, we will have to curate our own dataset.

The figure below shows a dozen of collected microalgae cell images in the `images/not_labeled/` folder.
{{< figure_resizing src="dataset_sample.png" >}}

There is only one issue now, and that is the images are not annotated. 
We will have to annotate all the images with an open source image labeling tool known as [labelImg](https://github.com/tzutalin/labelImg).


`labelImg` app enables us to annotate any image with class labels and bounding boxes surrounding the object of interest.
The following figure shows a demo of `labelImg` taken from the GitHub repository.
{{< figure_resizing src="labelimg_demo.jpg" >}}

Type `labelImg` in your terminal to launch the `labelImg` app.
A window like the following should appear.
{{< figure_resizing src="labelimg_start.png" >}}

Now, let's load the folder that contains the microalgae images into `labelImg` and annotate them! 
To do that, click on the **Open Dir** icon and navigate to the folder containing the images at `images/not_labeled/`. 
An image should now show up in `labelImg`.
Next click on the **Create RectBox** icon to start drawing bounding boxes around the microalgaes. Next you will be prompted to enter a label name. 
Key in microalgae as the label name. Once done, a rectangular bounding box should appear on-screen.

{{< figure_resizing src="labelimg_loaded.png" >}}

Now comes the repetitive part, we will need to draw a bounding box for each microalgae cell for all images in the folder.
To accelerate the process I highly recommend the use of Hotkeys keys with `labelImg`.
{{< figure_resizing src="hotkeys.png" width=400 >}}

Once done, remember to save the annotations. The annotations are saved in `XML` file with a filename matching to image as shown below.
{{< figure_resizing src="xml_files.png" >}}

Once all images are labelled, we will partition the image and annotations into three sets namely train set, validation set and test set.
These will be used to train and evaluate our model in the next section.

It took a few hours to meticulously label the images. 
The labeled images can be found in the `images/labeled/` folder in case you didn't want to label them as I did.

### 🏃 Developing a model
Once the labeling is done, we are now ready to train a model.
The training will be done in a `jupyter` notebook environment.

To launch jupyter notebook run 
```bash
jupyter lab
``` 

A browser window should pop up.
On the left pane, double click the `train.ipynb` to open the notebook.

The first cell in the notebook is the import.
In IceVision, just by running one line of code 

```python
from icevision.all import *
```

imports all the necessary packages for training.
If something wasn't properly installed, the imports will raise an error message.

#### Preparing datasets
We will now need to parse the images and its corresponding labels and bounding boxes with the following line.
```python
parser = parsers.VOCBBoxParser(annotations_dir="images/labeled", images_dir="images/labeled")
```

The argument `annotations_dir` and `images_dir` are the directory to the images and annotations respectively.
Since we had both the images and annotations in the same directory, they are the same as specified in the code.

Next we will divide the images and bounding boxes into two groups of data namely `train_records` and `valid_records`.
By default, the split will be `80:20` to `train:valid` proportion.
You can change the ratio by altering the value in `RandomSplitter`.

```python
train_records, valid_records = parser.parse(data_splitter=RandomSplitter([0.8, 0.2])
```

Running
```python
parser.class_map
```
outputs 
```python
<ClassMap: {'background': 0, 'Microalgae': 1}>
```
which shows the `ClassMap` that contains the class name as the key and class number as the value.
The `background` class is automatically added. We do not need to label the background.

#### Choosing library, model and backbone
Setting hyperparameters 

Load pre-trained model and train

Monitoring with callbacks

#### Training, monitoring and exporting model

### 🧭 Inferencing on a new image
Inference on a local machine
{{< figure_resizing src="inference.png" >}}

Figure illustrates the raw detection of cells from microscope image. The model is a RetinaNet with a ResNet50 backbone trained using [IceVision](https://github.com/airctic/icevision).
{{< figure_resizing src="detection.png" >}}