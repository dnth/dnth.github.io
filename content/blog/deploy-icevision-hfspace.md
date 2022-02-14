---
title: "Deploy IceVision Models on HuggingFace Spaces"
date: 2022-01-13T13:42:56+08:00
featureImage: images/blog/deploy-icevision-hfspace/thumb-train-deploy-share.png
postImage: images/blog/deploy-icevision-hfspace/train-deploy-share.png
---

{{< toc >}} 

### Introduction
So, you’ve trained a deep learning model that can detect objects from images. 
Next, how can you share the awesomeness of your model with the rest of the world? You might be a PhD student trying to get some ideas from your peers or supervisors, or a startup founder who wishes to share a minimum viable product to your clients for feedback. 
But at the same time do not wish to go through the hassle of dealing with MLOps. 
This blog post is for you. In this post I will walk you through how to deploy your model and share them to the world for free!

### Icevision
I will be using the awesome [IceVision](https://github.com/airctic/icevision) object detection package as an example for this post. 
IceVision is an agnostic computer vision library pluggable to multiple deep learning frameworks such as Fastai and PyTorch Lighting. 
What makes IceVision awesome is you can train a state-of-the-art object detection model with only few lines of codes. 
Check out the getting started tutorial [here](https://github.com/airctic/icevision/blob/master/notebooks/getting_started_object_detection.ipynb).

Upon completing the training of your model don't forget to save the model into a checkpoint to be used for inferencing later.
With IceVision this can be done easily. Add the following snippet to your notebook.

``` python
from icevision.models.checkpoint import *
save_icevision_checkpoint(learn.model,
                        model_name='torchvision.retinanet', 
                        backbone_name='resnet50_fpn',
                        img_size=image_size,
                        classes=parser.class_map.get_classes(),
                        filename='./models/model_checkpoint.pth',
                        meta={'icevision_version': '0.9.1'})
```


### Gradio
Next we will load the saved model checkpoint into Gradio that will provide a neat interface to the users who will use the app. 
IceVision repo provides a handy [notebook](https://github.com/airctic/icevision-gradio/blob/master/IceApp_coco.ipynb) that shows you how to deploy a trained model on Gradio. 
This should create a local and public link that can be accessed up to 72 hours as long as the notebook is kept open.
You can then share the link to anyone who would like to try out the app.

### HuggingFace Spaces
What if you would like the link to persist longer? One option is to deploy the Gradio app onto a free platform knwon as [HuggingFace Spaces](https://huggingface.co/spaces).
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

```python
import subprocess
import sys
print("Reinstalling mmcv")
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "mmcv-full==1.3.17"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "mmcv-full==1.3.17", "-f", "https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html"])
print("mmcv install complete") 

## Only works if we reinstall mmcv here.

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

for root, dirs, files in os.walk(r"sample_images/"):
    for filename in files:
        print("Loading sample image:", filename)


# Populate examples in Gradio interface
example_images = [["sample_images/" + file] for file in files]
# Columns: Input Image | Label | Box | Detection Threshold
examples = [
    [example_images[0], False, True, 0.5],
    [example_images[1], True, True, 0.5],
    [example_images[2], False, True, 0.7],
    [example_images[3], True, True, 0.7],
    [example_images[4], False, True, 0.5],
    [example_images[5], False, True, 0.5],
    [example_images[6], False, True, 0.6],
    [example_images[7], False, True, 0.6],
]


def show_preds(input_image, display_label, display_bbox, detection_threshold):
    if detection_threshold == 0:
        detection_threshold = 0.5
    img = PIL.Image.fromarray(input_image, "RGB")
    pred_dict = model_type.end2end_detect(
        img,
        valid_tfms,
        model,
        class_map=class_map,
        detection_threshold=detection_threshold,
        display_label=display_label,
        display_bbox=display_bbox,
        return_img=True,
        font_size=16,
        label_color="#FF59D6",
    )
    return pred_dict["img"], len(pred_dict["detection"]["bboxes"])


# display_chkbox = gr.inputs.CheckboxGroup(["Label", "BBox"], label="Display", default=True)
display_chkbox_label = gr.inputs.Checkbox(label="Label", default=False)
display_chkbox_box = gr.inputs.Checkbox(label="Box", default=True)
detection_threshold_slider = gr.inputs.Slider(
    minimum=0, maximum=1, step=0.1, default=0.5, label="Detection Threshold"
)
outputs = [
    gr.outputs.Image(type="pil", label="RetinaNet Inference"),
    gr.outputs.Textbox(type="number", label="Microalgae Count"),
]

article = "<p style='text-align: center'><a href='https://dicksonneoh.com/' target='_blank'>Blog post</a></p>"

# Option 1: Get an image from local drive
gr_interface = gr.Interface(
    fn=show_preds,
    inputs=[
        "image",
        display_chkbox_label,
        display_chkbox_box,
        detection_threshold_slider,
    ],
    outputs=outputs,
    title="Microalgae Detector with RetinaNet",
    description="This RetinaNet model counts microalgaes on a given image. Upload an image or click an example image below to use.",
    article=article,
    examples=examples,
)
# #  Option 2: Grab an image from a webcam
# gr_interface = gr.Interface(fn=show_preds, inputs=["webcam", display_chkbox_label, display_chkbox_box,  detection_threshold_slider], outputs=outputs, title='IceApp - COCO', live=False)
# #  Option 3: Continuous image stream from the webcam
# gr_interface = gr.Interface(fn=show_preds, inputs=["webcam", display_chkbox_label, display_chkbox_box,  detection_threshold_slider], outputs=outputs, title='IceApp - COCO', live=True)
gr_interface.launch(inline=False, share=False, debug=True)
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



