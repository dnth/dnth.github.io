---
title: "I Made It to GitHub Trending - My Open Source Journey"
date: 2024-11-04T20:48:15+08:00
featureImage: images/blog/i_made_it_to_github_trending/feature_image.gif
postImage: images/blog/i_made_it_to_github_trending/post_image.png
tags: ["open-source", "github-trending", "computer-vision", ]
categories: ["learning-in-public"]
toc: true
socialshare: true
description: "I made it to GitHub Trending! Here's how I did it."
images : 
- images/blog/i_made_it_to_github_trending/post_image.png
---


### 👋 Introduction

On October 28th, 2024 I made it to GitHub Trending! This wasn't something I expected to happen, and I'm still in disbelief.

{{< figure_autoresize src="trending_developer.png" width="auto" align="center" caption="GitHub Trending Developer for 28th October 2024" >}}

But after all the dopamine rush, I'm back to reality and I want to share my journey on how I made it. 

This is mostly a note to myself and I hope it can help you too.

### 🚀 The Project
I created an open source repository called [x.infer](https://github.com/dnth/x.infer). 

It's a framework agnostic computer vision inference library. 
I created this to solve my problem of using difference computer vision frameworks without the hassle of rewriting the inference code.

Check out the project here
{{< githubRepoCard "dnth/x.infer" >}}

With x.infer you can load any model from different frameworks and use them with a unified API.

```python
import xinfer

model = xinfer.create_model("vikhyatk/moondream2")
model.infer(image, prompt)         # Run single inference
model.infer_batch(images, prompts) # Run batch inference
model.launch_gradio()              # Launch Gradio interface
```

It already has support for many popular models and frameworks like Transformers, vLLM, Ollama, Ultralytics, etc. Combined it's over 1000 models that you can use with the above 4 lines of code.

{{< figure_autoresize src="code_typing.gif" width="500" align="center" >}}


### 🛣️ The Road to Trending

First a disclaimer, I think that building a project with the goal of getting it trending is NOT a good starting point. 

Anchoring your project on an extrinsic reward (getting trending) will not make you happier in the long run and you'll likely to give up when things are tough.

Instead, start with a goal that is much bigger. 

{{% blockquote %}}
Start with a goal to solve a problem, learn something, or build something that you're passionate about.
This will push you to work on your project for a longer without getting discouraged.
{{% /blockquote %}}

Now let's talk about how I did it.

### 🎯 Solving a Pain Point
The first step to getting your project trending is to solve a pain point. This is not something that you'll know immediately. 

But over time, you'll start to notice some patterns.
For me, one pattern emerged, there are many frameworks, and each framework has its own way of loading the model, inference, and post-processing, etc. 

Every time I change framework, I have to rewrite the inference code. This is really annoying.

Putting this into a single unified API would be super useful.

### 📝 The README File
This is probably the most important part of the project aside from solving a pain point.

Nobody would know about what pain point you're solving if you don't tell them in a way that's easy to understand.

I spent a good amount of time writing the README file. I wanted to make sure that it's easy to understand and that it showcases the project's value proposition.

At the very top, I included 4 things:
1. Shields.io badges to showcase the project's popularity and features.
2. Simple logo and a short description of what the project does.
3. A GIF to showcase the project's value proposition.
4. Quick links.

{{< figure_autoresize src="readme_breakdown.png" width="auto" align="center" >}}

This is the first thing people see when they visit the project. It's important to make it good. Like it or not, people judge a book by its cover. So make it good.

This also conveys that you're serious about your project.

### 🌟 Key Features

Most people are busy, don't make them read a long paragraph to convey something simple. Some better alternatives.

1. Bullet points
2. GIFs
3. Videos

For example, I used a GIF to showcase the project's value proposition.

{{< figure_autoresize src="https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/flowchart.gif" width="auto" align="center" >}}

And then follow up with a bullet point list to showcase the project's key features.

{{% blockquote %}}
✅ Run inference with >1000+ models in 3 lines of code.\
✅ List and search models interactively.\
✅ Launch a Gradio interface to interact with a model.\
✅ Serve model as a REST API endpoint with Ray Serve and FastAPI.\
✅ OpenAI chat completions API compatible.\
✅ Customize and add your own models with minimal code changes.
{{% /blockquote %}}



### 🎥 The Demo Video
To support the README file, I also created a demo video to showcase the project. This made it even clearer to understand what the project does. 

People are visual creatures they said and I find this to be true. And also visuals convey more information than words and in a shorter time.

{{< video src="gradio_demo.mp4" alt="Gradio Demo" autoplay="true" loop="true" >}}

Instead of trying to explain the project in words I let the video do the talking. Bonus point, the video is only 20 seconds but it conveys much more information than I could explain in words using a long paragraph. Plus, people are more likely to watch a video than to read a long paragraph.

### 👥 The Community

I did not post about this project on social media in one go. Instead I posted about it in a niche community to get some initial feedback.

I posted on Reddit and got some good feedback.

<blockquote class="reddit-embed-bq" style="height:316px" data-embed-locale="en-EN" data-embed-height="316"><a href="https://www.reddit.com/r/computervision/comments/1gbmuum/xinfer_framework_agnostic_computer_vision/">x.infer - Framework agnostic computer vision inference.</a><br> by<a href="https://www.reddit.com/user/WatercressTraining/">u/WatercressTraining</a> in<a href="https://www.reddit.com/r/computervision/">computervision</a></blockquote><script async="" src="https://embed.reddit.com/widgets.js" charset="UTF-8"></script>

</br>

### 🎁 Wrapping Up

There you have it. These are the steps I took to get my project trending. Hope you'll find it useful.

Once again, I want to emphasize that building a project with the goal of getting it trending is NOT a good starting point. 

Start with a goal to solve a problem, learn something, or build something that you're passionate about.
This will push you to work on your project for a longer without getting discouraged.

Lastly here's a [link to my project](https://github.com/dnth/x.infer) again in case you missed it. Drop a star if you like it!

{{< githubRepoCard "dnth/x.infer" >}}

Thank you for reading!
