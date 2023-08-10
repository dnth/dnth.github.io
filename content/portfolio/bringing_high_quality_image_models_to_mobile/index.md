---
title: "Bringing High-Quality Image Models to Mobile: Hugging Face TIMM Meets Android & iOS"
date: 2023-03-16T11:00:15+08:00
featureImage: images/portfolio/bringing_high_quality_image_models_to_mobile/thumbnail.gif
postImage: images/portfolio/bringing_high_quality_image_models_to_mobile/post_image.gif
tags: ["TIMM", "Hugging Face", "paddy-disease", "Flutter", "Android", "iOS", "EdgeNeXt", "Gradio"]
categories: ["deployment", "image-classification", "cloud-ai"]
toc: true
socialshare: true
description: "Discover how Hugging Face's TIMM library brings state-of-the-art computer vision models to iOS and Android."
images : 
- images/portfolio/bringing_high_quality_image_models_to_mobile/post_image.gif
---

### 🌟 Motivation
For many data scientist (including myself), we pride ourselves in training a model, seeing the loss graph go down, and claim victory when the test set accuracy reaches 99.99235%. 

Why not?

This is the after all the juiciest part of the job. "Solving" one dataset after another, it may seem like anything around you can be *conquered* with a simple `model.fit`.

That was me two years ago.

The naive version of me thought that was all about it with machine learning (ML).
As long as we have a dataset, ML is the way to go.

Almost nobody talked about what happens to the model after that.


{{< figure_resizing src="meme.jpg" >}}

{{% blockquote author="ChatGPT"%}}
Like a painting not shown in an artist's studio, a machine learning model not deployed is a missed opportunity to enrich and enhance the lives of those it was intended to serve.
{{% /blockquote %}}

<!-- So what is deployment anyway?

In simple terms, it is taking the trained model (that you took pride on) and integrating it into a system that can be used in real-world applications.

Broadly speaking, there are two deployment strategies in use today:
+ **Cloud-based Deployment** - This involves putting your model onto a cloud-based infrastructure (like AWS or Hugging Face) and running the inference on the cloud.
+ **Edge Deployment** - This involves putting your model on a device (like a mobile phone) and running the inference on the device.

Each has its pros and cons. So choose based on your use case.

Cloud-based deployment scales well. With the right infrastructure, your model can run thousands of inferences a second to serve millions of people. -->

Without deployment the model you've trained only benefits you.

So how do we maximize the number of people you can serve with the model?

Mobile device.

It's 2023, if you're reading this, chances are you own a mobile device.

<!-- [Interesting facts](https://techjury.net/blog/mobile-vs-desktop-usage/) -
+ In 2023, there are an estimated 16.8 billion mobile devices and counting!
+ 92.1% of internet users access the Web through mobile devices.
+ American adults spend an average of 5.5 hours daily on their cell phones in 2022. -->

Hands down, having a model that can work on mobile is going to reach many.

+ **Accessibility** - Most people carry their mobile phones with them. A model accessible on mobile devices lets users use models on the go.
+ **Built-in hardware** - Mobile devices comes packaged with on board camera and various sensors. Not worry about integrating new hardware.
+ **User experience** - Enables new form of interaction between apps and sensors on the phone. E.g. computer vision models can be used in an image editing app on the phone.

In this blog post, I will show you how you can make a model accessible through your mobile phone with Hugging Face and Flutter.

✅ Yes, for free.

{{< notice tip >}}
⚡ By the end of this post you will learn how to:
+ Upload a state-of-the-art image classification model to Hugging Face Spaces and get an inference endpoint.
+ Create a Flutter mobile app that runs on **Android** and **iOS** to call the inference endpoint.
+ Display the inference results on the screen with a beautiful UI.

💡 **NOTE**: Code and data for this post are available on my GitHub repo [here](https://github.com/dnth/huggingface-timm-mobile-blogpost).
{{< /notice >}}

Demo on iOS iPhone 14 Pro

![demo on ios](demo_ios.gif)

Demo on Android - Google Pixel 3 XL.

![demo on android](demo_android.gif)

I've also uploaded the app to Google Playstore. Download and try it out [here](https://play.google.com/store/apps/details?id=com.rice.net).

If that looks interesting, let's start!


### 🤗 Hugging Face x TIMM

Making computer vision models (especially large ones) available on mobile devices sounds interesting in theory.

But in practice there are many hurdles -

+ **Hardware limitation** - Mobile devices usually run on portable hardware with limited processing power, RAM, and battery life. Models needs to be optimized and efficient catering to these limitations.
+ **Optimization** - To put computer vision models on mobile hardware, they usually need to be optimized to run on specific hardware and software environment on the device. This requires specialized knowledge in computer vision and mobile development. 
+ **Practicality** - User experience is a big factor in whether your app will be used by anyone. Nobody wants to use a bloated, slow and inefficient mobile app.

I know that sounds complicated. 
Don't worry because we are **NOT** going to deal with any of that in this blog post!

Enter 👇


[Hugging Face](https://huggingface.co/) is a platform that allows users to host and share machine learning models and dataset. It's most notable for its Transformers model for Natural Language Processing (NLP).

Recently Hugging Face has been expanding its territory beyond NLP and venturing into computer vision. 

Ross Wightman, the creator of the wildly popular PyTorch Image Model (TIMM) repo joins forces. 

TIMM is a open-source computer vision repo used in research and commercial application. I boasts close to a thousand (and counting) state-of-the-art PyTorch image models, pretrained weights and scripts for training, validation and inference.



{{< figure_resizing src="hf_timm.png" caption="TIMM joins Hugging Face." >}}

{{< notice tip >}}
Check out the TIMM repo [here](https://github.com/huggingface/pytorch-image-models).
{{< /notice >}}

What does it mean for you?

Now you can use any models from TIMM with Hugging Face on platforms of your choice.
The Hugging Face docs shows [how you can do it using Python](https://huggingface.co/docs/hub/timm). 



<!-- - Introducing Hugging Face and TIMM as a solution
- Introduction to the TIMM (Timm Image Models) library and its architecture
- Advantages of using Hugging Face TIMM for mobile computer vision applications
- Introduction to the Hugging Face Model Hub and its collection of pretrained models
- Explanation of how to use pretrained models with Hugging Face TIMM for mobile computer vision applications
- Comparison of the performance of pretrained models with custom models on mobile devices
- Advantages and disadvantages of using pretrained models -->


### 📥 Hosting a Model on Hugging Face Spaces

Spaces are one of the most popular ways to share ML applications and demos with the world.

Details on the Hardware specifications and pricing [here](https://huggingface.co/pricing#spaces).

{{< figure_resizing src="spaces_specs.png" caption="Hardware specs on Spaces." >}}

{{< notice tip >}}
Details on how I trained the model using fastai [here](../pytorch_at_the_edge_timm_torchscript_flutter/#-training-with-fastai).
{{< /notice >}}

Here's the model that I trained using Fastai ahd hosted on Hugging Face Space. 


Try it out 👇
<iframe
	src="https://dnth-edgenext-paddy-disease-classifier.hf.space"
	frameborder="1"
	width="900"
	height="800"
></iframe>


{{< notice tip >}}
View on the Hugging Face webpage [here](https://dnth-edgenext-paddy-disease-classifier.hf.space).
{{< /notice >}}


The inference endpoind is deployed using Gradio in just a few lines of code.

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

{{< figure_resizing src="api_endpoint.png" caption="TIMM joins Hugging Face." >}}

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
+ Upload a SOTA classification model to Hugging Face Spaces and get an inference endpoint.
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
- Recap of Hugging Face TIMM's role in bringing these models to mobile devices
- Discussion of future possibilities for mobile computer vision using Hugging Face TIMM and other emerging technologies


### ⛄ FAQs
- What is computer vision, and why is it important for mobile applications?
- What is Hugging Face, and how does it relate to computer vision?
- What is the TIMM library, and what makes it unique compared to other computer vision libraries?
- What are the limitations of Android and iOS for computer vision applications?
- What is the Android Neural Networks API (NNAPI), and how does it work with Hugging Face TIMM?
- What is the Core ML framework, and how does it work with Hugging Face TIMM?
- What are pretrained models, and why are they important in computer vision?
- How do I use pretrained models with Hugging Face TIMM for mobile computer vision applications?
- How does the performance of pretrained models compare to custom models on mobile devices?
- What are the advantages and disadvantages of using pretrained models for mobile computer vision applications? -->