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
+ Set up a Telegram bot with the botfather.
+ Calling the Gradio API to access the GPT-J model prediction.
+ Hosting the Telegram bot on Hugging Face.
{{< /notice >}}



### 🤖 Setting up a Telegram Bot
Get token from botfather.

https://t.me/botfather


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
