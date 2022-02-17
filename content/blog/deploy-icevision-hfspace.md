---
title: "Deploy IceVision Models on HuggingFace Spaces"
date: 2022-02-17T13:42:56+08:00
featureImage: images/blog/deploy-icevision-hfspace/feature_image.gif
postImage: images/blog/deploy-icevision-hfspace/train-deploy-share.png
---

### Introduction
So, you’ve trained a deep learning model that can detect objects from images. 
Next, how can you share the awesomeness of your model with the rest of the world? 
You might be a PhD student trying to get some ideas from your peers or supervisors, or a startup founder who wishes to share a minimum viable product to your clients for feedback. 
But, at the same time you don't wish to go through the hassle of dealing with MLOps. 
This blog post is for you. In this post I will walk you through how to deploy your model and share them to the world for free!

### Training a Model with IceVision
We will be using the awesome [IceVision](https://github.com/airctic/icevision) object detection package as an example for this post. 
IceVision is an agnostic computer vision library pluggable to multiple deep learning frameworks such as [Fastai](https://github.com/fastai/fastai) and [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning). 
What makes IceVision awesome is you can train state-of-the-art object detection models with only few lines of codes.
It's very easy to get started, check out the tutorial [here](https://github.com/airctic/icevision/blob/master/notebooks/getting_started_object_detection.ipynb).

In the getting started notebook, we used a dataset from [Icedata](https://github.com/airctic/icedata) repository known as the *Fridge Objects* dataset.
This dataset consists 134 images of 4 classes: *can*, *carton*, *milk bottle*, *water bottle*.
Let's now continue to train our model. Let's train a *VFNet* model with a *ResNet* backbone implemented in [mmdet](https://github.com/open-mmlab/mmdetection).
In the notebook, you can easily specify this model using two lines of codes as follows.

```python
model_type = models.mmdet.vfnet
backbone = model_type.backbones.resnet50_fpn_mstrain_2x
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
save_icevision_checkpoint(model,
                        model_name='mmdet.vfnet', 
                        backbone_name='resnet50_fpn_mstrain_2x',
                        img_size=image_size,
                        classes=parser.class_map.get_classes(),
                        filename='./models/model_checkpoint.pth',
                        meta={'icevision_version': '0.12.0'})
```
The notebook that I used for this section can be found [here](https://colab.research.google.com/github/dnth/dnth.github.io/blob/main/static/images/blog/deploy-icevision-hfspace/training_vfnet.ipynb).

### User Interface with Gradio
At this point, in order to run inference on the model, one will need to write inference codes as shown [here](https://airctic.com/0.12.0/).
This is non-trivial and can be time-consuming especially to those who are not familiar.
Gradio simplifies this by providing a simple graphical user interface wrapping the inference code so that anyone can run inference on the model without having to code.

The following figure shows a screenshot of the Gradio app that runs in the browser.
The left pane shows the input image, and the right pane shows the inference results.
Users can upload an image or select from a list of example images and click on *Submit* to run it through the model for inference.

{{< figure src="/images/blog/deploy-icevision-hfspace/gradio.png" alt="Screenshot of the Onion homepage" width=750 >}}

So how do we load our model into the Gradio app? First, we must first install the Gradio package by running `pip install gradio`.
Next, create a file with the name `app.py` and paste the following codes into the file.
```python
from gradio.outputs import Label
from icevision.all import *
from icevision.models.checkpoint import *
import PIL
import gradio as gr
import os

# Load model
checkpoint_path = "model_checkpoint.pth"
checkpoint_and_model = model_from_checkpoint(checkpoint_path)
model = checkpoint_and_model["model"]
model_type = checkpoint_and_model["model_type"]
class_map = checkpoint_and_model["class_map"]

# Transforms
img_size = checkpoint_and_model["img_size"]
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(img_size), tfms.A.Normalize()])

# Populate examples in Gradio interface
examples = [
    ['1.jpg'],
    ['2.jpg'],
    ['3.jpg']
]

def show_preds(input_image):
    img = PIL.Image.fromarray(input_image, "RGB")
    pred_dict = model_type.end2end_detect(img, valid_tfms, model, 
                                          class_map=class_map, 
                                          detection_threshold=0.5,
                                          display_label=False, 
                                          display_bbox=True, 
                                          return_img=True, 
                                          font_size=16, 
                                          label_color="#FF59D6")
    return pred_dict["img"]

gr_interface = gr.Interface(
    fn=show_preds,
    inputs=["image"],
    outputs=[gr.outputs.Image(type="pil", label="RetinaNet Inference")],
    title="Fridge Object Detector",
    description="A VFNet model that detects common objects found in fridge. Upload an image or click an example image below to use.",
    examples=examples,
)
gr_interface.launch(inline=False, share=False, debug=True)
```

Make sure that the path to your model checkpoint and example images are correct.
Running the `app.py` loads our model into the Gradio app.
Run the script by typing `python app.py` in the terminal.
If there are no errors, the terminal will show local URL to access the Gradio app.
You can copy the address and open it with a browser.
The URL address on my machine is `http://127.0.0.1:7860/`, it may vary on yours.

### HuggingFace Spaces
The Gradio app URL link from the previous section can only be accessed locally. But what if you would like to share the link to someone across the internet for free?
In this section, we will discover how to make your Gradio app accessible to anyone by deploying the app on a free platform known as HuggingFace [Spaces](https://huggingface.co/spaces).
Spaces is the new 'marketplace' for various bleeding edge machine learning models.
Many researchers have uploaded interesting and state-of-the-art models on Space to showcase them as a demo.
You can discover and try them out [here](https://huggingface.co/spaces).

#### Creating a Space
To host a model on Spaces, you must sign up for an account at [`https://huggingface.co/`](https://huggingface.co/).
After that, head over to [`https://huggingface.co/spaces`](https://huggingface.co/spaces) and click on **Create New Space** button as shown below.

{{< figure src="/images/blog/deploy-icevision-hfspace/create_new_space.png" alt="Screenshot of the Onion homepage" width=750 >}}

Next fill in the Space name and select a License. 
Make sure to select Gradio as the Space SDK and keep the repository **Public**. Click on **Create space** button when you're done.

{{< figure src="/images/blog/deploy-icevision-hfspace/space_details.png" alt="Screenshot of the Onion homepage" width=750 >}}

Once done, your Space is now ready.
The Space you've created behaves like a `git` repository.
You can perform various `git` related operations such as `git clone`, `git push` and `git pull` to update the repository.
Alternatively, you can also add files into the Space directly in the browser.

{{< figure src="/images/blog/deploy-icevision-hfspace/empty_repo.png" alt="Screenshot of the Onion homepage" width=750 >}}
In this blog post, I am going to show you how add files into your Space using the browser. 

#### Installation files
Let's start with the installation files. These are the files that determines the packages that will be installed on your Space to run your app.
The packages are specified in two files ie. `requirements.txt`, and `packages.txt`.

The `requirements.txt` lists all the `Python` packages that will be `pip`-installed on the Space.
The `packages.txt` is a file created to specify the OpenCV version to be installed on your Space.
For some reason putting the OpenCV package in the `requirements.txt` file doesn't work.

Let's begin adding these files.
Click on the **Files and versions** tab. Next, click on **Add file** and **Create a new file**.
{{< figure src="/images/blog/deploy-icevision-hfspace/files_version_tab.png" alt="Screenshot of the Onion homepage" width=750 >}}

Name your file as `requirements.txt` and paste the following snippets as the content. Click on **Commit new file** button at the bottom of the page.
```bash
--find-links https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html
mmcv-full==1.3.17
mmdet==2.17.0
gradio==2.7.5
icevision[all]==0.12.0
```

Now, do the same for `packages.txt` which only has the OpenCV package as the file content.
```bash
python3-opencv
```
We are now done adding all installation files into our Space.


#### Gradio app, checkpoint and sample images
Next let's add the Gradio app, model checkpoint and some sample images.
Let's add the `app.py` we had from the previous section using the same method we did for the installation files.
The `app.py` hosts the logic of your application and this is where the code for the Gradio app resides.
Space will automatically run `app.py` upon startup.


Next let's add our checkpoint file `model_checkpoint.pth` by clicking on **Upload File**.
Drag and drop the model checkpoint file and click on **Commit changes**.

<!-- {{< figure src="/images/blog/deploy-icevision-hfspace/upload_file.png" alt="Screenshot of the Onion homepage" width=750 >}} -->
{{< figure src="/images/blog/deploy-icevision-hfspace/drop_files.png" alt="Screenshot of the Onion homepage" width=750 >}}

You will also see a **Building** status indicating that it is setting up by installing the packages and running it upon completion.
Every time there is a change in any of the files, the Space will be rebuilt.

Using the same method let's upload the sample images as well.
At this point the Space repository should look like the following
{{< figure src="/images/blog/deploy-icevision-hfspace/complete_upload.png" alt="Screenshot of the Onion homepage" width=750 >}}
Once the building completes, the status changes to **Running** and the Space should look like the following and is now ready to be used.

{{< figure src="/images/blog/deploy-icevision-hfspace/screenshot_apps.png" alt="Screenshot of the Onion homepage" width=750 >}}

The final app looks like the following
{{< figure src="/images/blog/deploy-icevision-hfspace/screenshot_final.png" alt="Screenshot of the Onion homepage" width=750 >}}
You can now share the URL to your Space to anyone across the internet for free.
[Here](https://huggingface.co/spaces/dnth/icevision_fridge_tutorial) is the link to the Space used in this blog post.


<!-- <html>
<head>
<link rel="stylesheet" href="https://gradio.s3-us-west-2.amazonaws.com/2.6.2/static/bundle.css">
</head>
<body>
<div id="target"></div>
<script src="https://gradio.s3-us-west-2.amazonaws.com/2.6.2/static/bundle.js"></script>
<script>
launchGradioFromSpaces("dnth/webdemo-fridge-detection", "#target")
</script>
</body>
</html> -->
