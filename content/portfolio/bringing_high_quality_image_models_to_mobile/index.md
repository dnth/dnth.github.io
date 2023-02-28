---
title: "Bringing High-Quality Image Models to Mobile: HuggingFace TIMM Meets Android & iOS"
date: 2022-02-27T11:00:15+08:00
featureImage: images/portfolio/bringing_high_quality_image_models_to_mobile/thumbnail.gif
postImage: images/portfolio/bringing_high_quality_image_models_to_mobile/post_image.gif
tags: ["TIMM", "HuggingFace", "paddy-disease", "Flutter", "Android", "iOS", "EdgeNeXt"]
categories: ["deployment", "image-classification", "cloud-ai"]
toc: true
socialshare: true
description: "Discover how HuggingFace's TIMM library enables integration of state-of-the-art computer vision models on mobile."
images : 
- images/portfolio/bringing_high_quality_image_models_to_mobile/post_image.gif
---

{{< notice info >}}
This blog post is still a work in progress. If you require further clarifications before the contents are finalized, please get in touch with me [here](https://dicksonneoh.com/contact/), on [LinkedIn](https://www.linkedin.com/in/dickson-neoh/), or [Twitter](https://twitter.com/dicksonneoh7).
{{< /notice >}}


### 🌟 Motivation

Meet Bob, a data scientist with a passion for computer vision. Bob had been working on a project to build a model that could identify different types of fruits, from apples to pineapples. He spent countless hours training and fine-tuning the model until it could recognize fruits with 98% accuracy.

Bob even bagged few hackathon awards and praised my many to be the "expert" in computer vision.

Bob was me 5 years ago.

Before joining the industry, I was an academic and researcher in a university in Malaysia.

My favorite part of the job? You've guessed it. Hitting `model.train` on SOTA models and publish them on "reputable journals" just to wash my hands clean and repeat.

{{< figure_resizing src="meme.jpg" >}}

I realized that it doesn't matter how many papers publish about SOTA models, it would not change anything it stays on paper. 

{{% blockquote author="ChatGPT"%}}
Like a painting that remains unseen in an artist's studio, a machine learning model that remains undeployed is a missed opportunity to enrich and enhance the lives of those it was intended to serve.
{{% /blockquote %}}

Why would anyone want to deploy models on mobile devices? 

Here are a few reasons -
+ **Accessibility** - Most people carry their mobile phones with them. A model accessible on mobile devices lets users use models on the go.
+ **Built-in hardware** - Mobile devices comes packaged with on board camera and various sensors. Not worry about integrating new hardware.
+ **User experience** - Enables new form of interaction between apps and sensors on the phone. E.g. computer vision models can be used in an image editing app on the phone.

✅ Yes, for free.

{{< notice tip >}}
⚡ By the end of this post you will learn how to:
+ Upload a SOTA classification model to HuggingFace Spaces and get an inference endpoint.
+ Create a Flutter mobile app that runs on **Android** and **iOS** to call the inference endpoint.
+ Display the inference results on the screen with a beautiful UI.

💡 **NOTE**: Code and data for this post are available on my GitHub repo [here](https://github.com/dnth/huggingface-timm-mobile-blogpost).
{{< /notice >}}

Demo on iOS iPhone 14 Pro

![demo on ios](demo_ios.gif)

Demo on Android - Google Pixel 3 XL.

![demo on android](demo_android.gif)

I've also uploaded the app to Google Playstore. Download and try it out [here](https://play.google.com/store/apps/details?id=com.rice.net).

Making computer vision models (especially large ones) available on mobile devices sounds interesting in theory.

But in practice there are many hurdles -

+ **Hardware limitation** - Mobile devices usually run on portable hardware with limited processing power, RAM, and battery life. Models needs to be optimized and efficient catering to these limitations.
+ **Optimization** - To put computer vision models on mobile hardware, they usually need to be optimized to run on specific hardware and software environment on the device. This requires specialized knowledge in computer vision and mobile development. 
+ **Practicality** - User experience is a big factor in whether your app will be used by anyone. Nobody wants to use a bloated, slow and inefficient mobile app.

I know that sounds complicated. 
Don't worry because we are **NOT** going to deal with any of that in this blog post!

Enter 👇


### 🤗 HuggingFace x TIMM
[HuggingFace](https://huggingface.co/) is a platform that allows users to host and share machine learning models and dataset. It's most notable for its Transformers model for Natural Language Processing (NLP).

Recently HuggingFace has been expanding its territory beyond NLP and venturing into computer vision. 

Ross Wightman, the creator of the wildly popular PyTorch Image Model (TIMM) repo joins forces. 

TIMM is a open-source computer vision repo used in research and commercial application. I boasts close to a thousand (and counting) state-of-the-art PyTorch image models, pretrained weights and scripts for training, validation and inference.



{{< figure_resizing src="hf_timm.png" caption="TIMM joins HuggingFace." >}}

{{< notice tip >}}


Check out the TIMM repo [here](https://github.com/huggingface/pytorch-image-models).
{{< /notice >}}

What does it mean for you?

Now you can use any models from TIMM with HuggingFace on platforms of your choice.
The HuggingFace docs shows [how you can do it using Python](https://huggingface.co/docs/hub/timm). 



<!-- - Introducing HuggingFace and TIMM as a solution
- Introduction to the TIMM (Timm Image Models) library and its architecture
- Advantages of using HuggingFace TIMM for mobile computer vision applications
- Introduction to the HuggingFace Model Hub and its collection of pretrained models
- Explanation of how to use pretrained models with HuggingFace TIMM for mobile computer vision applications
- Comparison of the performance of pretrained models with custom models on mobile devices
- Advantages and disadvantages of using pretrained models -->


### 📥 Hosting a Model on HuggingFace Spaces

Spaces are one of the most popular ways to share ML applications and demos with the world.

Hardware specs [here](https://huggingface.co/pricing#spaces).

{{< figure_resizing src="spaces_specs.png" caption="Hardware specs on Spaces." >}}


Details on how I trained the model is [here](../pytorch_at_the_edge_timm_torchscript_flutter/#-training-with-fastai).

Here's the model that I trained using Fastai ahd hosted on HuggingFace Space. 


Try it out 👇
<iframe
	src="https://dnth-edgenext-paddy-disease-classifie-dc60651.hf.space"
	frameborder="1"
	width="900"
	height="800"
></iframe>

View on the HuggingFace webpage [here](https://dnth-edgenext-paddy-disease-classifie-dc60651.hf.space).

Deployed using Gradio.

```python {linenos=table}
import os
import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner("learner.pkl")
labels = learn.dls.vocab

def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Paddy Disease Classifier with EdgeNeXt"
description = "9 Diseases + 1 Normal class"
interpretation = "default"
examples = ["sample_images/" + file for file in files]
enable_queue = True

gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(shape=(224, 224)),
    outputs=gr.outputs.Label(num_top_classes=3),
    title=title,
    description=description,
    examples=examples,
    interpretation=interpretation,
    enable_queue=enable_queue
).launch()
```

If we want to use other language then we'll need an API endpoint.


### 🔄 Inference API Endpoint
All applications deployed using Gradio has an API endpoint.

{{< figure_resizing src="api_endpoint.png" caption="TIMM joins HuggingFace." >}}

View the API endpoint [here](https://dnth-edgenext-paddy-disease-classifie-dc60651.hf.space/?view=api)

### 📲 Flutter
- Build user interface.
- Don't want to get user lost in the detail implementation.



Calling the endpoint in `Flutter`.

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

Future<Map> classifyRiceImage(String imageBase64) async {
  final response = await http.post(
    Uri.parse(
        'https://dnth-edgenext-paddy-disease-classifie-dc60651.hf.space/run/predict'),
    headers: <String, String>{
      'Content-Type': 'application/json; charset=UTF-8',
    },
    body: jsonEncode(<String, List<String>>{
      'data': [imageBase64]
    }),
  );

  if (response.statusCode == 200) {
    // If the server did return a 200 CREATED response,
    // then decode the image and return it.
    final classificationResult = jsonDecode(response.body)["data"][0];
    return classificationResult;
  } else {
    // If the server did not return a 200 OKAY response,
    // then throw an exception.
    throw Exception('Failed to classify image.');
  }
}
```

GitHub repo [here](https://github.com/dnth/huggingface-timm-mobile-blogpost/tree/main/lib).

### 🤖 Demo

Demo on iOS iPhone 14 Pro

![demo on ios](demo_ios.gif)

Demo on Android - Google Pixel 3 XL.

![demo on android](demo_android.gif)

Use image picker or camera.

I've also uploaded the app to Google Playstore. Download and try it out [here](https://play.google.com/store/apps/details?id=com.rice.net).

### 🙏 Comments & Feedback
That's a wrap! In this post, I've shown you how you can start from a model, train it, and deploy it on a mobile device for edge inference.

{{< notice tip >}}
⚡ By the end of this post you will learn how to:
+ Upload a SOTA classification model to HuggingFace Spaces and get an inference endpoint.
+ Create a Flutter mobile app that runs on **Android** and **iOS** to call the inference endpoint.
+ Display the inference results on the screen with a beautiful UI.

💡 **NOTE**: Code and data for this post are available on my GitHub repo [here](https://github.com/dnth/huggingface-timm-mobile-blogpost).
{{< /notice >}}
What's next? If you'd like to learn about how I deploy a cloud based object detection model on Android, check it out [here](../how_to_deploy_od_models_on_android_with_flutter/).



I hope you've learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter/LinkedIn post or [drop me a message](https://dicksonneoh.com/contact/). 

<!-- Alternatively you can also comment on this Hacker News [thread](https://news.ycombinator.com/item?id=34799597#34801672). -->

<!-- {{< tweet dicksonneoh7 1625367344712388609>}}


<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:ugcPost:7032246822186209280" height="1198" width="504" onload='javascript:(function(o){o.style.height=o.contentWindow.document.body.scrollHeight+"px";}(this));' frameborder="0" allowfullscreen="" title="Embedded post"></iframe> -->


<!-- ### 🎄 Conclusion

- Summary of the importance of high-quality image models for mobile applications
- Recap of HuggingFace TIMM's role in bringing these models to mobile devices
- Discussion of future possibilities for mobile computer vision using HuggingFace TIMM and other emerging technologies


### ⛄ FAQs
- What is computer vision, and why is it important for mobile applications?
- What is HuggingFace, and how does it relate to computer vision?
- What is the TIMM library, and what makes it unique compared to other computer vision libraries?
- What are the limitations of Android and iOS for computer vision applications?
- What is the Android Neural Networks API (NNAPI), and how does it work with HuggingFace TIMM?
- What is the Core ML framework, and how does it work with HuggingFace TIMM?
- What are pretrained models, and why are they important in computer vision?
- How do I use pretrained models with HuggingFace TIMM for mobile computer vision applications?
- How does the performance of pretrained models compare to custom models on mobile devices?
- What are the advantages and disadvantages of using pretrained models for mobile computer vision applications? -->