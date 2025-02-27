---
title: "How to Deploy Object Detection Models on Android with Flutter"
date: 2022-04-17T15:00:15+08:00
featureImage: images/portfolio/how_to_deploy_od_models_on_android_with_flutter/thumbnail.gif
postImage: images/portfolio/how_to_deploy_od_models_on_android_with_flutter/post_image.png
tags: ["Flutter", "Hugging-Face", "Android", "Fast.ai", "Gradio", "IceVision"]
categories: ["deployment", "object-detection", "cloud-ai"]
toc: true
socialshare: true
description: "Leverage giant models in the cloud on Android with Hugging Face and Flutter."
images : 
- images/portfolio/how_to_deploy_od_models_on_android_with_flutter/post_image.png
---

### 🚑 Deployment: Where ML models go to die

In this post, I will outline the basic steps to deploy ML models onto lightweight mobile devices **easily, quickly and for free**.

{{< notice tip >}}
By the end of this post, you will learn about:

* Leveraging Hugging Face infrastructure to host models.
* Deploying on any edge device using REST API.
* Displaying the results on a Flutter Android app.
{{< /notice >}}

According to [Gartner](https://www.gartner.com/en/newsroom/press-releases/2018-02-13-gartner-says-nearly-half-of-cios-are-planning-to-deploy-artificial-intelligence), more than **85%** of machine learning (ML) models never made it into production.
This trend is expected to continue further this year in 2022.

In other words, despite all the promises and hype around ML, most models fail to deliver in a production environment.
According to Barr Moses, CEO, Monte Carlo, [deployment](https://towardsdatascience.com/why-production-machine-learning-fails-and-how-to-fix-it-b59616184604) is one of the critical points where many models fail.

So what exactly is the deployment of ML model? 
Simply put, deployment is making a model's capability or insight available to other users or systems - [Luigi Patruno](https://mlinproduction.com/what-does-it-mean-to-deploy-a-machine-learning-model-deployment-series-01/).


### 🏹 Begin with deployment in mind
Many ML researchers take pride in training bleeding-edge models with state-of-the-art (SOTA) results on datasets.
As a researcher, I understand how deeply satisfying it feels to accomplish that.

Unfortunately, many of these so-called "SOTA models" will end up on preprints, Jupyter notebooks, or in some obscure repository, nobody cares about after the initial hype. 

Eventually, they are forgotten and lost in the ocean of newer "SOTA models".
To make things worse, the obsession with chasing after "SOTA models" often causes researchers to lose track of the end goal of building the model - **deployment**.

<!-- We can forget about ever finding these models in production. -->

{{< figure_autoresize src="jupyter_meme.png" caption="Source: ClearML on Reddit." link="https://www.reddit.com/r/mlops/comments/o8w2e4/you_know_the_deal_if_you_dont_post_content_ill/?utm_source=share&utm_medium=ios_app&utm_name=iossmf">}}

Hence, as ML engineers, it is very helpful if we build models with deployment in mind, as the end result.

Because only when a model is deployed can it add value to businesses or organizations.
This is the beginning of getting a model into production. 

*Deployment* is unfortunately a messy and complicated topic in [MLOps](https://databricks.com/glossary/mlops) - too deep for us to cover here. Luckily, that is not the purpose of this post.

{{< notice tip >}}
My objective in this post is to show you how you can deploy an ML model easily on a mobile device without getting your hands dirty with servers, backends or Kubernetes.
{{< /notice >}}

<!-- Once the model is built, we can immediately spin up an interactive demo.
An interactive demo opens the door to users' feedbacks from using the model which are invaluable in product iteration to prepare for further stages.

Unfortunately, many don't even make it through this phase.
This is not anyone's fault, as making an interactive demo often requires skills beyond ML. -->

<!-- In this post I'm going to show you that is no longer the case.

**Anyone with no knowledge about backend, servers, or Kubernetes can quickly spin up an interactive demo, deploy them on the cloud or on a mobile device and share it to users to gain feedback.** -->

The following figure shows the deployment architecture that allows us to accomplish that.
{{< figure_autoresize src="architecture.png" caption="Deployment architecture.">}}

### 🤗 Hosting a model on Hugging Face
The first piece of the puzzle is to host our model on some cloud infrastructure.
In this post, let's use a free service known as Hugging Face *Spaces*.

*Spaces* is a platform where anyone can upload their model and share it with the world.
If you head to https://huggingface.co/spaces, you will find thousands of models that researchers made freely available online.
{{< figure_autoresize src="spaces_web.png">}}

These models are hosted on *Spaces* for demo and sharing purposes. 
But they can be scaled up into full-fledge production with the [Inference API](https://huggingface.co/inference-api).

Let's set up a Space to host our model. If you're unsure how to do that, I wrote a recent guide on how to set your own Space with the Gradio app [here](https://dicksonneoh.com/portfolio/deploy_icevision_models_on_huggingface_spaces/).

In this post, I will use an IceVision object detection model trained to detect microalgae cells from an image.
I trained this model in under a minute with only 17 labeled images. [Here](https://dicksonneoh.com/portfolio/training_dl_model_for_cell_counting/) is how I did it.

Once the Space is set, we will have a Gradio interface like the following
{{< figure src="space_demo.gif" width=750 >}}

This Space is now ready to be shared with anyone with an internet connection and a browser.
Try the live demo below 👇👇👇

<iframe src="https://hf.space/embed/dnth/webdemo-microalgae-counting/+" frameBorder="0" width="1000" height="1000" title="Gradio app" class="container p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

But what if we want to make the app work on a mobile device **without using a browser?** Enter 👇

### 📞 Calling the HTTP Endpoint
One neat feature of the Gradio app is it exposes the model through a RESTful API.
This makes the model prediction accessible via HTTP request which we can conveniently use on any mobile device!

Now, any computationally lightweight device can make use of the model's prediction just by running a simple HTTP call.
All the heavy lifting is taken care of by the Hugging Face infrastructure. 

{{< notice tip >}}
This can be a game-changer if the model is complex and the edge device is not powerful enough to run the model - which is a common scenario.
{{< /notice >}}

Additionally, this also reduces deployment hardware costs, because now any lightweight, portable mobile device with an internet connection can leverage the model's cell counting capability.

The figure below shows the endpoint for us to call the model.

{{< figure_autoresize src="api_endpoint.png">}}

As shown, the input to the model is an image, and the output, an image (with bounding boxes) and also a value of the microalgae count. You can check out the API [here](https://hf.space/embed/dnth/webdemo-microalgae-counting/api).

If you'd like to test the HTTP endpoint live, head to the API [page](https://hf.space/embed/dnth/webdemo-microalgae-counting/api) as the following figure.
{{< figure_autoresize src="test_endpoint.png">}}

Alternatively, you can also try them out on your computer with `curl`:

```bash
curl -X POST https://hf.space/embed/dnth/webdemo-microalgae-counting/+/api/predict/ 
-H 'Content-Type: application/json' 
-d '{"data": ["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHB..."]}'
```

Let's copy the `URL endpoint` and use in the next section



### 📲 Displaying results in Flutter
We will be using Flutter to make a simple Android app that sends an image and receive the bounding box prediction via HTTP calls.


Flutter uses the *Dart* programming language that makes it incredibly easy to construct a graphical user interface (GUI).
I omit the codes to construct the GUI in this post for simplicity. 
Let me know if you'd like to access it. 
There are also tons of tutorials on how to construct the GUI so, I will not cover them here too.

The snippet of code that allows us to perform the HTTP call to the Hugging Face server is as follows.
```dart {linenos=table}
import 'dart:convert';
import 'package:http/http.dart' as http;

Future<Map> detectImage(String imageBase64) async {
  final response = await http.post(
    Uri.parse(
        'https://hf.space/gradioiframe/dnth/webdemo-microalgae-counting/+/api/predict/'),
    headers: <String, String>{
      'Content-Type': 'application/json; charset=UTF-8',
    },
    body: jsonEncode(<String, List<dynamic>>{
      'data': [imageBase64]
    }),
  );

  if (response.statusCode == 200) {
    final detectionResult = jsonDecode(response.body)["data"];

    final imageData =
        detectionResult[0].replaceAll('data:image/png;base64,', '');

    return {"count": detectionResult[1], "image": imageData};
    // If the server did return a 200 CREATED response,
    // then decode the image and return it.
  } else {
    // If the server did not return a 200 OKAY response,
    // then throw an exception.
    throw Exception('Failed to get results.');
  }
}
```

The `detectImage` function in `line 4` takes in a single parameter `String` `base64` format image and returns a `Map` which consists of the image with bounding box and the microalgae count in `line 22`. 

The `URL endpoint` that we copied from the previous section is on `line 7`.

The screen recording below illustrates the Flutter app sending a sample image to the Hugging Face inference server and getting a response on the number of detected microalgae cells and the image with all the bounding boxes.


<!-- {{< figure src="microsense.gif" width=500 >}}
{{< youtube DLmyG-K8lAw >}} -->

{{< video src="algae1.webm" width="600px" loop="true" autoplay="true" muted="true" >}}


I published the app on Google Playstore.
If you like, try them out [here]((https://play.google.com/store/apps/details?id=com.micro.sense)). 

I've also published another similar app that deploys a deep learning classifier model (trained with Fastai) that categorizes paddy leaf diseases [here](https://play.google.com/store/apps/details?id=com.rice.net) using the same approach outlined in this post.

### 💡 Up Next
That's about it! In this post hopefully, it's clear now that deploying deep learning models on mobile devices doesn't need to be complicated - at least in the beginning when it's critical to gain users' feedback before deciding if it's right to scale up.

Caveat: I do acknowledge that the approach in this post might not be optimal in some circumstances, especially if you have thousands of users on your app.

For that, I would recommend scaling up to use the Hugging Face [Inference API](https://huggingface.co/inference-api) - a fully hosted production-ready solution 👇.
{{< figure_autoresize src="inference_api.png">}}

It is also possible now to deploy Hugging Face models on AWS Sagemaker for serverless inference. 
Check them out [here](https://aws.amazon.com/machine-learning/hugging-face/).

Finally, you could also use the same Flutter codebase and export it into an iOS, Windows, or even a Web app. 
This is the beauty of using Flutter for front-end development.
Code once, and export to multiple platforms.

### 🙏 Comments & Feedback
If you have any questions, comments, or feedback, please leave them on the following Twitter post or [drop me a message](https://dicksonneoh.com/contact/).
{{< tweet user="dicksonneoh7" id="1517004585495240704">}}

<!-- {{< notice info >}}
This blog post is still a work in progress. If you require further clarifications before the contents are finalized, please get in touch with me [here](https://dicksonneoh.com/contact/), on [LinkedIn](https://www.linkedin.com/in/dickson-neoh/), or [Twitter](https://twitter.com/dicksonneoh7).
{{< /notice >}} -->