---
title: "Training a Deep Learning Model for Cell Counting in 17 Lines of Code"
date: 2022-04-08T15:07:15+08:00
featureImage: images/blog/training_dl_model_for_cell_counting/thumbnail.gif
postImage: images/blog/training_dl_model_for_cell_counting/post_image.jpg
tags: ["IceVision", "Fast.ai", "counting", "cell"]
categories: ["modeling", "object-detection", "biology"]
toc: true
socialshare: true
description: "Get started in minutes by leveraging state-of-the-art object detection models on IceVision with Fastai"
images : 
- images/blog/training_dl_model_for_cell_counting/post_image.jpg
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
We will use a library known as [IceVision](https://airctic.com/0.12.0/) - a computer vision focused library built to work with [Fastai](https://github.com/fastai/fastai). 

Before we start, I highly recommend that you use a virtual environment system such as Anaconda. 
[Here](https://www.geeksforgeeks.org/set-up-virtual-environment-for-python-using-anaconda/) is how to set it up.

All the codes and data you will need to replicate this post is available on this Github [repository](https://github.com/dnth/microalgae-cell-counter-blogpost).
To get started, let's clone the Git repository by typing the following in your terminal:

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

Depending on your system `CUDA` version, you may want to change `cuda11` to `cuda10` especially on older systems. 
The number following the `CUDA` version is the version of IceVision. 
The version I'm using for this blog post is `0.12.0`.
You can alternatively specify `master` to install the bleeding edge version of IceVision from the master branch on Github.

If you would like to install the CPU version of the library it can be done with:
```bash
bash icevision_install.sh cpu 0.12.0
```
Be warned, training object detection models on a CPU can be many times slower on a CPU compared to a GPU.
If you do not have an available GPU, use [Google Colab](https://colab.research.google.com/).
They provide free GPU instances.

The installation may take a couple of minutes depending on the speed of your internet connection.
Allow the installation to complete before proceeding to the next step.


### 🔖 Labeling the data
Deep learning models require data to work.
In this post, we want to construct a model that can count microalgaes from images. 
For that, must have images of microalges cells to work with.
The figure below shows a dozen of collected microalgae cell images in the `data/not_labeled/` folder.
{{< figure_resizing src="dataset_sample.png" >}}

There is only one issue now, and that is the images are not labeled. 
Let's label the images with bounding boxes using open-source image labeling tool [labelImg](https://github.com/tzutalin/labelImg).



The `labelImg` app enables us to label images with class name and bounding boxes surrounding the object of interest.
The following figure shows a demo of `labelImg` app.
{{< figure_resizing src="labelimg_demo_annot.jpg" >}}

The `labelImg` is already installed in the installation step.
Now, you can type 
```bash
labelImg
``` 
in your terminal to launch the `labelImg` app.
A window like the following should appear.
{{< figure_resizing src="labelimg_start.png" >}}

Now, let's load the folder that contains the microalgae images into `labelImg` and annotate them! 
To do that, click on the **Open Dir** icon and navigate to the folder containing the images at `data/not_labeled/`. 
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
The labeled images can be found in the `data/labeled/` folder in case you didn't want to label them as I did.

### 🌀 Modeling
Once the labeling is done, we are now ready to start modeling.
The modeling will be done in a `jupyter` notebook environment.

To launch jupyter notebook run 
```bash
jupyter lab
``` 

A browser window should pop up.
On the left pane, double click the `train.ipynb` to open the notebook.

All the codes in this section are in the `train.ipynb` notebook.
Here, I will attempt to walk you through just enough details of the code to get you started with modeling on your own data.
If you require further clarifications, the IceVision [documentation](https://airctic.com/0.12.0/) is a good starting point.
Alternatively, [drop me a message](https://dicksonneoh.com/contact/).

The first cell in the notebook are the imports.
With IceVision all the necessary components are imported with one line of code

```python
from icevision.all import *
```


If something wasn't properly installed, the imports will raise an error message.
In that event, you must go back to the installation (Step 1) before proceeding further.
If there are no errors, we are ready to dive in further.


#### 🎯 Preparing datasets
After the imports, we must now read the labeled images and bounding boxes into the `jupyter` notebook.
This is also known as parsing the data and can be accomplished with the following

```python
parser = parsers.VOCBBoxParser(annotations_dir="data/labeled", images_dir="data/labeled")
```

The argument `annotations_dir` and `images_dir` are the directory to the images and annotations respectively.
Since we had both the images and annotations in the same directory, they are the same as specified in the code.

Next we will divide the images and bounding boxes into two groups of data namely `train_records` and `valid_records`.
By default, the split will be `80:20` to `train:valid` proportion.
You can change the ratio by altering the value in `RandomSplitter`.

```python
train_records, valid_records = parser.parse(data_splitter=RandomSplitter([0.8, 0.2])
```

If you would like to check what the class names from the parsed data try running
```python
parser.class_map
```

It should output 
```python
<ClassMap: {'background': 0, 'Microalgae': 1}>
```


which shows the `ClassMap` that contains the class name as the key and class number as the value in a Python [dictionary](https://www.w3schools.com/python/python_dictionaries.asp).
The `background` class is automatically added. 
In Step 2 we do not need to label the background.

Next, we will apply basic data augmentation which is a technique used to diversify the training images by applying random transformation.
Learn more [here](https://medium.com/analytics-vidhya/image-augmentation-9b7be3972e27).

The following code specifies the kinds of transformation we would like to perform on our images.
Behind the scenes these transformations are performed with the [Albumentations](https://albumentations.ai/) library.

```python
image_size = 640
train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=image_size+128), tfms.A.Normalize()])
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])
```

We must specify the dimensions of the image in `image_size = 640`. 
This value will then be used in `tfms.A.aug_tfms` that ensures that all images are resized to a `640x640` resolution and normalized in `tfms.A.Normalize()`.

Some models like `EfficientDet` only works with image size divisible by `128`.
Other common values you may try are `384`,`512`,`768`, etc. 
But beware using large image size may consume more memory and in some cases halts training.
Starting with a small value like `384` is probably a good idea.
I found that for this blog post `640` works best.

Use `tfms.A.aug_tfms` also performs other transformations to the image such as varying the lighting, rotation, shifting, flipping, blurring, padding, etc.
The full list of transforms that and the arguments can be found in the `aug_tfms` [documentation](https://airctic.com/0.12.0/albumentations_tfms/).

In this code snippet we created two distinct transforms namely `train_tfms` and `valid_tfms` that will be used during the training and validation steps.

Next, we will apply the `train_tfms` to our `train_records` and `valid_tfms` to `valid_records` with the following snippet.

```python
train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)
```

This results in the creation of a `Dataset` object which is a collection of transformed images and bounding boxes.

To visualize the `train_ds` we can run

```python
samples = [train_ds[0] for _ in range(4)]
show_samples(samples, ncols=4)
```

This will show us 4 samples from the `train_ds`.
Note the variations in lighting, translation, and rotation compared to the original images.
{{< figure_resizing src="show_ds.png" >}}

The transformations are applied on-the-fly.
So each run on the snippet produces different results.


#### 🗝️ Choosing library, model, and backbone
IceVision supports hundreds of high-quality pre-trained models from [Torchvision](https://github.com/pytorch/vision), Open MMLab's [MMDetection](https://github.com/open-mmlab/mmdetection), Ultralytic's [YOLOv5](https://github.com/ultralytics/yolov5) and Ross Wightman's [EfficientDet](https://github.com/rwightman/efficientdet-pytorch).

Depending on your preference, you may choose the model and backbone from these libraries.
In this post I will choose the [VarifocalNet](https://arxiv.org/abs/2008.13367) (VFNet) model from MMDetection which can be accomplished with

```python
model_type = models.mmdet.vfnet
backbone = model_type.backbones.resnet50_fpn_mstrain_2x
model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(parser.class_map)) 
```

There are various ResNet backbones that you can select from such as
`resnet50_fpn_1x`,
`resnet50_fpn_mstrain_2x`,
`resnet50_fpn_mdconv_c3_c5_mstrain_2x`,
`resnet101_fpn_1x`,
`resnet101_fpn_mstrain_2x`,
`resnet101_fpn_mdconv_c3_c5_mstrain_2x`,
`resnext101_32x4d_fpn_mdconv_c3_c5_mstrain_2x`, and
`resnext101_64x4d_fpn_mdconv_c3_c5_mstrain_2x`.

Additionally, IceVsion also recently supports state-of-the-art Swin Transformer backbone for the VFNet model
`swin_t_p4_w7_fpn_1x_coco`,
`swin_s_p4_w7_fpn_1x_coco`, and
`swin_b_p4_w7_fpn_1x_coco`.


Which combination of `model_type` and `backbone` that performs best is something you need to experiment with.
Feel free to experiment and swap out the backbone and note the performance of the model.

#### 🏃 Metrics and Training
In order to start the training, the model needs to ingest the images and bounding boxes from the `train_ds` and `valid_ds` we created.
This is the role that dataloaders play.

We will therefore need to construct the dataloaders from the `train_ds` and `valid_ds` respectively

```python
train_dl = model_type.train_dl(train_ds, batch_size=2, num_workers=4, shuffle=True)
valid_dl = model_type.valid_dl(valid_ds, batch_size=2, num_workers=4, shuffle=False)
```

Here, we can specify the `batch_size` which is the number of images and bounding boxes to be passed to the model in a single forward pass.
The `batch_size` is a hyperparameter that be tuned to improve performance.
The `num_workers` argument specifies the number of CPU cores to be used - the more cores, the faster.

Next, we need to specify the metric we use for the training. 
Metric is a measure of how good the model is at the task we are trying to train the model for.
Some commonly used metrics include accuracy, F1 Score, etc.
For object detection tasks the `COCOMetric` is commonly used.

One of the most important hyperparameter to get right is the learning rate.
Since IceVision is built to work with Fastai, we have access to a handy tool known as the learning rate finder first proposed by Leslie Smith and popularized by the Fastai community for its effectiveness.
This is an incredibly easy to use tool to find a range of optimal learning rate with this dataset.

All we need to do is run

```python
learn.lr_find()
```

which outputs
{{< figure_resizing src="lr_find.png" >}}

The most optimal learning rate value is where the loss descends most rapidly as can be seen in values between `1e-4` to `1e-3`.
The orange dot on the plot shows the point where the slope is the steepest.

With this learning rate value, we can pass it into the fine_tune function to start training.

```python
metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
learn = model_type.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics)
learn.fine_tune(10, 3e-4, freeze_epochs=1)
```

The first argument to in `fine_tune` is the number of epochs to train for. In this post I will train for 10 epochs for demonstration. 
Training for longer will likely to improve the model.
The second argument is the base learning rate value which we found using the `learn.lr_find()`

The above code snippet trains the model for 10 `epochs`.
The `freeze_epochs` specifies the number of `epochs` to train while the backbone of the model is frozen.

The figure below shows the training output.
In ➀, only the last layer of the model was trained.
The remaining parts of the model are frozen.
In ➁, the entire mode is trained end-to-end.

{{< figure_resizing src="train.png" >}}

During the training, the `train_loss`, `valid_loss` and `COCOMetric`is printed every epoch.
Ideally the losses should decrease and `COCOMetric` increase the longer we train.
As shown above, each epoch only took 2 seconds to complete on a GPU - which is incredibly fast.




```python
model_type.show_results(model, valid_ds, detection_threshold=.5)
```

{{< figure_resizing src="show_results.png" >}}

For completeness here are all the codes in Step 3 to load the data, instantiate the model, training and showing the results.
That is only 17 lines of code if you remove the spaces in between!

```python {linenos=table}
from icevision.all import *

parser = parsers.VOCBBoxParser(annotations_dir="data/labeled", images_dir="data/labeled")
train_records, valid_records = parser.parse()

image_size = 640
train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=image_size+128), tfms.A.Normalize()])
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])

train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

model_type = models.mmdet.vfnet
backbone = model_type.backbones.resnet50_fpn_mstrain_2x
model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(parser.class_map)) 

train_dl = model_type.train_dl(train_ds, batch_size=2, num_workers=4, shuffle=True)
valid_dl = model_type.valid_dl(valid_ds, batch_size=2, num_workers=4, shuffle=False)

metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
learn = model_type.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics)
learn.fine_tune(10, 3e-4, freeze_epochs=1)

model_type.show_results(model, valid_ds, detection_threshold=.5)
```

#### 📨 Exporting model
Once you are satisfied with the performance and quality of the model, you can export the model and save all its configurations into a `checkpoint` with

```python
from icevision.models.checkpoint import *
save_icevision_checkpoint(model,
                        model_name='mmdet.vfnet', 
                        backbone_name='resnet50_fpn_mstrain_2x',
                        img_size=640,
                        classes=parser.class_map.get_classes(),
                        filename='./models/model_checkpoint.pth',
                        meta={'icevision_version': '0.12.0'})
```
The arguments `model_name`, `backbone_name` and `img_size` has to match what we've used during training.
`filename` specifies the directory and name of the checkpoint file.
`meta` is an optional argument you can use to save all other information about the model.

### 🧭 Inferencing on a new image
```python
from icevision.all import *
from icevision.models.checkpoint import *
from PIL import Image
```


```python
checkpoint_path = "./models/model_checkpoint.pth"
checkpoint_and_model = model_from_checkpoint(checkpoint_path)
model = checkpoint_and_model["model"]
model_type = checkpoint_and_model["model_type"]
class_map = checkpoint_and_model["class_map"]
```

```python
img_size = checkpoint_and_model["img_size"]
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(img_size), tfms.A.Normalize()])
```

Inference on a local machine
{{< figure_resizing src="inference.png" >}}

Codes are in the `inference.ipynb` notebook.



```python
img = Image.open('data/not_labeled/IMG_20191203_164256.jpg')
```


```python
pred_dict = model_type.end2end_detect(img, valid_tfms, model, 
                                      class_map=class_map, 
                                      detection_threshold=0.5,
                                      display_label=True, 
                                      display_bbox=True, 
                                      return_img=True, 
                                      font_size=50, 
                                      label_color="#FF59D6")
```


```python
len(pred_dict['detection']['bboxes'])
```

```python
pred_dict["img"]
```

```python
pred_dict["img"].save("inference.png")
```

Figure illustrates the raw detection of cells from microscope image. The model is a RetinaNet with a ResNet50 backbone trained using [IceVision](https://github.com/airctic/icevision).
{{< figure_resizing src="inference.png" >}}