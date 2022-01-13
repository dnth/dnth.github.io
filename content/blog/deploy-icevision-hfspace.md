---
title: "Deploy Icevision Models on HuggingFace Spaces"
date: 2022-01-13T13:42:56+08:00
featureImage: images/allpost/allPost-7.jpg
postImage: images/blog/gradioxhf.png
---


### Introduction
So, you’ve done it. You finally trained a deep learning model that is capable of detecting objects from your dataset. Next, how can you share the awesomeness of your model with the rest of the world? You might be a PhD student trying to get some ideas from your peers or supervisors, or a startup founder who wishes to share a minimum viable product to your clients for feedback. But at the same time do not wish to go through the hassle of dealing with MLOps. This blog post is for you. In this post I will walk you through how easy it is to train a model and deploy them to the world for free.

### Icevision

I will be using the awesome Icevision library as an example for this post. However, the same concept applies even if you use other libraries. Icevision is an agnostic computer vision library pluggable to multiple deep learning frameworks such as Fastai and Pytorch Lighting. What makes Icevision awesome is you can train a state-of-the-art object detection model with ease. Check out their tutorial here to get started. In case you are interested here is the notebook taken from the Icevision tutorial that I used to train my model. Feel free to run the notebook in the Google Colab and modify them to your need.


### Gradio
Assuming you are done with training your object detection model with Icevision in the previous section, you will have a saved Pytorch model. Now we will need to load the model into a Gradio app. Gradio is an interface to deploying DL models using Python. Icevision provides a handy notebook that shows you how to deploy a trained model on Gradio. Check it out here. This should create a local and public link that can be accessed up to 72 hours. What if you would like the link to persist for more than that? We will need to deploy the app onto Hugging Face Space.





### HuggingFace Spaces
Before we can deploy the app on Hugging Face space, we need to specify what packages are required to run the inference. Therefore we need to create a text file `requirements.txt` that specifies a list of packages to be installed.
Hugging Face Spaces is a place that can host DL models for free. Good for showcasing a model. Many models available. We will use HF Space to host the Gradio app. To deploy the Gradio app we created on Hugging Face Spaces, we need to create a python script app.py. This script will be run when the app loads on Hugging Face Space.

![](/images/blog/screenshot.png )



