---
title: "Deploy IceVision Models on HuggingFace Spaces"
date: 2022-01-13T13:42:56+08:00
featureImage: images/blog/deploy-icevision-hfspace/thumb-train-deploy-share.png
postImage: images/blog/deploy-icevision-hfspace/train-deploy-share.png
---

{{< toc >}} 

### Introduction
So, you’ve trained a deep learning model that can detect objects from images. 
Next, how can you share the awesomeness of your model with the rest of the world? 
You might be a PhD student trying to get some ideas from your peers or supervisors, or a startup founder who wishes to share a minimum viable product to your clients for feedback. 
But, at the same time you don't wish to go through the hassle of dealing with MLOps. 
This blog post is for you. In this post I will walk you through how to deploy your model and share them to the world for free!

### Training a Model with IceVision
We will be using the awesome [IceVision](https://github.com/airctic/icevision) object detection package as an example for this post. 
IceVision is an agnostic computer vision library pluggable to multiple deep learning frameworks such as [Fastai](https://github.com/fastai/fastai) and [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning). 
What makes IceVision awesome is you can train a state-of-the-art object detection models with only few lines of codes.
It's very easy to get started, check out the tutorial [here](https://github.com/airctic/icevision/blob/master/notebooks/getting_started_object_detection.ipynb).

In the getting started notebook, we use a dataset from [Icedata](https://github.com/airctic/icedata) repository known as the *Fridge Objects* dataset.
This dataset consists 134 images of 4 classes: *can*, *carton*, *milk bottle*, *water bottle*.
Let's now continue to train our model. Let's train a simple *RetinaNet* model with a *ResNet* backbone from [Torchvision](https://github.com/pytorch/vision).
In the notebook, you can easily specify this model using two line of codes as follows.

```python
model_type = models.torchvision.retinanet
backbone = model_type.backbones.resnet50_fpn
```

After you're satisfied with the performance of your model, let's save the model into a checkpoint to be used for inferencing later.
With IceVision this can be done easily. Just add the following snippet to your notebook and run.
Feel free to modify the `model_name`, `backbone_name` according to the model you used during training.
The `img_size` argument is image size that the model is trained on.
The `classes` argument is a list of classes from the dataset.
The `filename` argument specifies the directory and name of the checkpoint file.
The `meta` argument stores other metadata that you would like to keep track of for future reference.

``` python
from icevision.models.checkpoint import *
save_icevision_checkpoint(learn.model,
                        model_name='torchvision.retinanet', 
                        backbone_name='resnet50_fpn',
                        img_size=image_size,
                        classes=parser.class_map.get_classes(),
                        filename='./models/model_checkpoint.pth',
                        meta={'icevision_version': '0.12.0'})
```
The notebook that I used for this section can be found [here](https://colab.research.google.com/github/dnth/dnth.github.io/blob/main/static/images/blog/deploy-icevision-hfspace/training_retinanet.ipynb).

### User Interface with Gradio
At this point, in order to run inference on the model, one will need to write inference codes as shown [here](https://airctic.com/0.12.0/).
This is non-trivial especially to people who don't code.
Gradio simplifies this by providing a simple user interface so that anyone can run an inference on the model without having to code.

The following figure shows a screenshot of the Gradio user interface that runs in the browser.
The left pane shows the input image.
User can select from a list of images and click on *Submit* to run it through the model for inference.
The result of the inference is shown on the right pane.

{{< figure src="/images/blog/deploy-icevision-hfspace/gradio.png" alt="Screenshot of the Onion homepage" width=750 >}}


In order to use Gradio, we must first install it with `pip install gradio`.
Next, the following Python script is used load our model into and load it into the Gradio user interface.
```python
from gradio.outputs import Label
from icevision.all import *
from icevision.models.checkpoint import *
import PIL
import gradio as gr
import os

# Load model
checkpoint_path = "models/model_checkpoint.pth"
checkpoint_and_model = model_from_checkpoint(checkpoint_path)
model = checkpoint_and_model["model"]
model_type = checkpoint_and_model["model_type"]
class_map = checkpoint_and_model["class_map"]

# Transforms
img_size = checkpoint_and_model["img_size"]
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(img_size), tfms.A.Normalize()])

# Populate examples in Gradio interface
examples = [['sample_images/1.jpg'],['sample_images/2.jpg'],['sample_images/3.jpg']]

def show_preds(input_image):
    img = PIL.Image.fromarray(input_image, "RGB")
    pred_dict = model_type.end2end_detect(img, valid_tfms, model, class_map=class_map, detection_threshold=0.5,
                                           display_label=False, display_bbox=True, return_img=True, 
                                           font_size=16, label_color="#FF59D6")
    return pred_dict["img"]

gr_interface = gr.Interface(
    fn=show_preds,
    inputs=["image"],
    outputs=[gr.outputs.Image(type="pil", label="RetinaNet Inference")],
    title="Fridge Object Detector",
    description="This RetinaNet model detects common objects found in fridge. Upload an image or click an example image below to use.",
    examples=examples,
)
gr_interface.launch(inline=False, share=False, debug=True)
```

Upon successful running of the Python script the model should be available for inference in the Gradio app at a local URL accessible with a browser, for instance `http://127.0.0.1:7860/`.


### HuggingFace Spaces
This local URL link from the previous section can only be accessed locally. What if you would like to share the link to someone across the internet?
In this section, we will discover how to make the Gradio app accessible to anyone by deploying the Gradio app onto a free platform known as HuggingFace [Spaces](https://huggingface.co/spaces).
Spaces is the new marketplace for all various bleeding edge of machine learning models.
Models hosted on Spaces are free for access at anytime.

#### Creating a Space
To host a model on Spaces, you must sign-up for an account.
After that head over to [`https://huggingface.co/spaces`](https://huggingface.co/spaces) and click on **Create New Space** button.

{{< figure src="/images/blog/deploy-icevision-hfspace/create_new_space.png" alt="Screenshot of the Onion homepage" width=750 >}}

Next fill in the details of the `Space`. Make sure to select `Gradio` as the `Space SDK` and keep the repository **Public**. Click on Create space button when you're done.

{{< figure src="/images/blog/deploy-icevision-hfspace/space_details.png" alt="Screenshot of the Onion homepage" width=750 >}}

Once done, your `Space` is now ready to be used.
The `Space` you've created behaves like a `git` repository.
You can perform various `git` related operations such as `git clone`, `git push` and `git pull` to update the repository.
Alternatively, you can also add files into the Space directly using the user interface.

{{< figure src="/images/blog/deploy-icevision-hfspace/empty_repo.png" alt="Screenshot of the Onion homepage" width=750 >}}

#### Adding related files
In this post, I am going to show you how to do it via the user interface. 
Click on the Files and versions tab.
You can now begin adding the files here.

{{< figure src="/images/blog/deploy-icevision-hfspace/files_version_tab.png" alt="Screenshot of the Onion homepage" width=750 >}}

There are a few files required to setup the `Space` namely `app.py`, `requirements.txt`, and `packages.txt`.

`app.py` hosts the logic of your application. This is where the code for the Gradio interface resides.
This script will be run when the app loads on Hugging Face Space.

`requirements.txt` lists all the `Python` packages that will be `pip`-installed on the `Space`.

`packages.txt` is special file created to put the OpenCV package to make it work on Spaces. 

This is the contennt of app.py

If you used a `mmdetection` model add the following lines at the beginning of the `app.py`.

```python
import subprocess
import sys
print("Reinstalling mmcv")
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "mmcv-full==1.3.17"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "mmcv-full==1.3.17", "-f", "https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html"])
print("mmcv install complete") 
```



Content of `requirements.txt`
```
mmdet==2.19.0
gradio==2.4.0
icevision[all]
mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html
```

Content of `packages.txt`
```
python3-opencv
```




{{< figure src="/images/blog/deploy-icevision-hfspace/screenshot_1.png" alt="Screenshot of the Onion homepage" width=750 >}}



