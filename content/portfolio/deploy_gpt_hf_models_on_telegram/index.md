---
title: "Deploying GPT-J Model on a Telegram Bot with Hugging Face Spaces"
date: 2022-01-11T11:00:15+08:00
featureImage: images/portfolio/deploy_gpt_hf_models_on_telegram/thumbnail.gif
postImage: images/portfolio/deploy_gpt_hf_models_on_telegram/post_image.png
tags: ["GPT-J", "Gradio", "Hugging Face", "Telegram"]
categories: ["deployment", "NLP"]
toc: true
socialshare: true
description: "Monitor your models with Wandb and pick the best!"
images : 
- images/portfolio/deploy_gpt_hf_models_on_telegram/post_image.png
---

{{< notice info >}}
This blog post is still a work in progress. If you require further clarifications before the contents are finalized, please get in touch with me [here](https://dicksonneoh.com/contact/), on [LinkedIn](https://www.linkedin.com/in/dickson-neoh/), or [Twitter](https://twitter.com/dicksonneoh7).
{{< /notice >}}

### 💥 Motivation

{{< notice tip >}}
By the end of this post you will learn how to:
+ Set up a Telegram bot with a `Python` wrapper library. 
+ Use the Gradio API to access the GPT-J model prediction.
+ Host the Telegram bot on Hugging Face `Spaces`.
{{< /notice >}}

Deploying a state-of-the-art (SOTA) GPT language model on a chatbot can be tricky.
You might wonder how to gain access the GPT model? On which infrastructure should you host the bot and the model? Serverless? AWS? Kubernetes?

Yada.. yada.. yada..

Things get complicated easily and I get it. It's definitely not worth going down that rabbit hole if you're only experimenting and toying around.

{{< figure_resizing src="gpt-aws-kubernetes.jpg">}}

In this post I will show you how I deploy a SOTA GPT-J model by [EleutherAI](https://www.eleuther.ai/) on a Telegram bot for free. 

By the end of this blog post you'll have your very own Telegram bot that can query the GPT-J model with any text you send it 👇👇👇

{{< video src="chatbot.mp4" width="400px" loop="true" autoplay="true" muted="true">}}


If that looks interesting, let's begin 👩‍💻


### 🤖 Setting up a Telegram Bot
First, we need to set up a Telegram bot that is associated with your Telegram account.
If you don't have a Telegram account, you can [create](https://telegram.org/) one for free.

Once already have an account, click [here](https://t.me/botfather) to start creating a bot.
Alternatively, you can go to the Telegram search bar and search for `botfather`.

{{< figure_resizing src="botfather.jpg" width=400 >}}

Next, send `/start` to the `botfather` to start a conversation.
Follow the instruction given in the botfather chat until you obtain a **token** for your bot.

{{< notice warning >}}

Keep this token private. Anyone with this token has access to your bot.

{{< /notice >}}


This video provides a good step-by-step visual guide on how to obtain a token for your bot.
{{< youtube aNmRNjME6mE >}}


### 💡 Creating a Gradio App
Every Gradio interface comes with an API that you can use to access the functions within.

Use an availble space on https://huggingface.co/spaces/akhaliq/gpt-j-6B

We will send `POST` request to access the GPT-J model prediction.

```python
def get_gpt_response(text):
    r = requests.post(
        url="https://hf.space/embed/akhaliq/gpt-j-6B/+/api/predict/",
        json={"data": [text]},
    )
    response = r.json()
    return response["data"][0]
```

Documentation on the Gradio API [here](https://www.gradio.app/using_the_api_docs/).


### 🤗 Hosting on Hugging Face Spaces
Set up API key as secrets.

https://huggingface.co/spaces/dnth/ptb-gpt

### 🎉 Conclusion

Link to Telegram bot
https://t.me/ptbgptbot


### 🙏 Comments & Feedback
I hope you've learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter post or [drop me a message](https://dicksonneoh.com/contact/).
{{< tweet dicksonneoh7 1524263583097384960>}}


If you like what you see and don't want to miss any of my future contents, follow me on Twitter and LinkedIn where I deliver more of these tips in bite-size posts.