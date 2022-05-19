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

*Yada.. yada.. yada..*

Things get complicated easily and I get it. It's definitely not worth going down that rabbit hole if you're only experimenting and toying around.

{{< figure_resizing src="gpt-aws-kubernetes.jpg">}}

In this post I will show you how I deploy a SOTA GPT-J model by [EleutherAI](https://www.eleuther.ai/) on a Telegram bot for free. 

By the end of this blog post you'll have your very own Telegram bot that can query the GPT-J model with any text you send it 👇👇👇

{{< video src="chatbot.mp4" width="400px" loop="true" autoplay="true" muted="true">}}


If that looks interesting, let's begin 👩‍💻


### 🤖 Token From the Mighty BotFather
{{< figure_resizing src="botfather_img.png" width=400 >}}
*We shall start by first appeasing the mighty `BotFather` who holds the key to the world of bots* 🤖

If you don't have a Telegram account, you must first [create](https://telegram.org/) one. It's free.

Next, we need to set up a bot that is associated with your Telegram account.
For that, let's consult the `BotFather` and initiate the bot creation. The first result is the `BotFather`.

This [link](https://t.me/botfather) brings you to the `BotFather`.
Alternatively, type `BotFather` in the Telegram search bar.

{{< figure_resizing src="botfather.jpg" width=400 >}}

Next, send `/start` to the `BotFather` to start a conversation.
Follow the instructions given by the `BotFather` until you obtain a **token** for your bot.

{{< notice warning >}}
Keep this **token** private. Anyone with this **token** has access to your bot.
{{< /notice >}}


This video provides a good step-by-step visual guide on how to obtain a **token** from the `BotFather`.
{{< youtube aNmRNjME6mE >}}


### 🐍 Python Telegram Bot



Telegram wasn't written with `Python`.
But we ❤️ `Python`!
Can we still use `Python` to code our bot?

Yes! ✅ With a wrapper library like [`python-telegram-bot`](https://github.com/python-telegram-bot/python-telegram-bot).
{{< figure_resizing src="ptb-logo.png" link="https://github.com/python-telegram-bot/python-telegram-bot" >}}

`python-telegram-bot` provides a pure `Python`, asynchronous interface for the [Telegram Bot API](https://core.telegram.org/bots/api).
It is also incredibly user-friendly and easy to start.
You can start running your own Telegram bot with only 8 lines of code 👇

```python {linenos=table}
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

def hello(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(f'Hello {update.effective_user.first_name}.')

updater = Updater('YOUR-TOKEN')
updater.dispatcher.add_handler(CommandHandler('start', hello))
updater.start_polling()
updater.idle()
```

The above code snippet creates a Telegram bot that recognizes the `/start` command (specified on `line 8`).
Upon receiving the `/start` command it calls the `hello` function on `line 4` which replies to the user.

Here's a screen recording showing that 👇 

{{< video src="start.mp4" width="400px" loop="true" autoplay="true" muted="true">}}

Yes! It's that simple! 🤓

Now all you have to do is specify other commands to call any other functions of your choice.

Before we do that, let's first install `python-telegram-bot` via

```bash
pip install python-telegram-bot==13.11
```

{{< notice warning >}}
`python-telegram-bot` is under active development. There are breaking changes starting version `20` and onward. For this post, I'd recommend sticking with version `<20`.
{{< /notice >}}

To run the bot, save the 8-line code snippet above into a `.py` file and run it on your computer. 
Remember to replace `'YOUR-TOKEN'` on `line 7` with your own **token** from the `BotFather`.

I will save the codes as `bot.py` on my machine and run the script with

```python
python bot.py
```

*Voila!*

Your bot is now live and ready to chat.
Search for your bot on the Telegram search bar, and send it the `/start` command.
It should respond by replying a text back to you, just like in the screen recording above.



### 💡 GPT-J and the Gradio API
We've configured our Telegram bot. 
What about the GPT model? 
Unless you have a powerful computer that runs 24/7, I wouldn't recommend running the GPT model on your machine (although you can).

I recently found a better solution which you can use to host the GPT model. 
Anyone can use it, it runs 24/7, and best of all it's free!

Enter 👉 [Hugging Face Hub](https://huggingface.co/docs/hub/main).

Hugging Face Hub is a central place where anyone can share their models, dataset and app demos.
The 3 main repo types of the Hugging Face Hub include:
+ Models - hosts models.
+ Datasets - stores datasets.
+ Spaces - hosts demo apps.

The GPT-J-6B model is generously provided by EleutherAI on the Hugging Face Hub as a model repository.
It's publicly available for use. Check them out [here](https://huggingface.co/EleutherAI/gpt-j-6B).

You can interact with the model directly on the GPT-J-6B model repo, or create a demo on your Space.
In this post, I will show you how to set up a Gradio app on Hugging Face Space to interact with the GPT-J-6B model.

First create a Space with your Hugging Face account.
If you're unsure how to do that, I wrote a guide [here](https://dicksonneoh.com/portfolio/deploy_icevision_models_on_huggingface_spaces/#hugging-face-spaces).
Next, add the `app.py` file to run this Space.

It looks like the following 👇

```python {linenos=table}
import gradio as gr

title = "GPT-J-6B"

description = "Gradio Demo for GPT-J 6B, a transformer model trained \
using Ben Wang's Mesh Transformer JAX. 'GPT-J' refers to the class of \
model, while '6B' represents the number of trainable parameters. \
To use it, simply add your text, or click one of the examples to load them. \
I've used the API on this Space to deploy the GPT-J-6B model on a Telegram bot. \
Link to blog post below 👇"

article = "<p style='text-align: center'> \
<a href='https://dicksonneoh.com/portfolio/deploy_gpt_hf_models_on_telegram/' \
target='_blank'>Blog post</a></p>"

examples = [
    ['The tower is 324 metres (1,063 ft) tall,'],
    ["The Moon's orbit around Earth has"],
    ["The smooth Borealis basin in the Northern Hemisphere covers 40%"]
]

gr.Interface.load("huggingface/EleutherAI/gpt-j-6B", 
                inputs=gr.inputs.Textbox(lines=5, label="Input Text"),
                title=title,description=description,
                article=article, 
                examples=examples,
                enable_queue=True).launch()

```

On `line 22` we are loading the GPT model directly from the [EleutherAI model hub](https://huggingface.co/EleutherAI) and serving the predictions on the Space.

Once the build completes, your Space is live.
Check out the running demo app on my [Space](https://huggingface.co/spaces/dnth/gpt-j-6B).
Or try them out 👇

<iframe src="https://hf.space/embed/dnth/gpt-j-6B/+" frameBorder="0" width="800" height="900" title="Gradio app" class="container p-0 flex-grow space-iframe" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>

A Gradio app comes with an API endpoint that you can use to access the app from elsewhere. For example, I've used this feature to get model predictions on my Android app [here](https://dicksonneoh.com/portfolio/how_to_deploy_od_models_on_android_with_flutter/).

To view the API, click on "view the api" button at the bottom of the Space.
It brings you to the API [page](https://hf.space/embed/dnth/gpt-j-6B/api) that shows you how to use the endpoint.

All we need to do now is send a `POST` request from our Telegram bot to access the GPT-J model prediction.

```python
def get_gpt_response(text):
    r = requests.post(
        url="https://hf.space/embed/dnth/gpt-j-6B/+/api/predict/",
        json={"data": [text]},
    )
    response = r.json()
    return response["data"][0]
```

Let's add this function into the `bot.py` file we created earlier.
Here's mine

```python {linenos=table}
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters
import requests
from telegram import ChatAction
import os

def hello(update: Update, context: CallbackContext) -> None:
    intro_text = """
    🤖 Greetings human! \n
    🤗 I'm a bot hosted on Hugging Face Spaces. \n
    🦾 I can query the mighty GPT-J-6B model and send you a response here. Try me.\n
    ✉️ Send me a text to start and I shall generate a response to complete your text!\n\n
    ‼️ PS: Responses are not my own (everything's from GPT-J-6B). I'm not conscious (yet).\n
    Blog post: https://dicksonneoh.com/portfolio/deploy_gpt_hf_models_on_telegram/
    """
    update.message.reply_text(intro_text)

def get_gpt_response(text):
    r = requests.post(
        url="https://hf.space/embed/dnth/gpt-j-6B/+/api/predict/",
        json={"data": [text]},
    )
    response = r.json()
    return response["data"][0]

def respond_to_user(update: Update, context: CallbackContext):
    update.message.chat.send_action(action=ChatAction.TYPING)
    response_text = get_gpt_response(update.message.text)
    update.message.reply_text(response_text)

updater = Updater('YOUR-TOKEN')
updater.dispatcher.add_handler(CommandHandler("start", hello))
updater.dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, respond_to_user))
updater.start_polling()
updater.idle()
```

I'm gonna save this as `app.py` on my computer and run it via

```bash
python app.py
```

Now, your bot will respond to `/start` command by calling the `hello` function.
And additionally, it will respond to all non-command texts by calling the `respond_to_user` function (Configured on `line 33`).

That is how we get GPT-J's response via the Telegram bot 🤖.
If you've made it to this point, congratulations! We're almost done!

{{< notice tip>}}
If you wish to run the Telegram bot on your machine you can stop here.
Bear in mind you need to keep your machine alive 24/7 for your bot to work.
{{< /notice >}}

But, if you wish to take your bot to the next level 🚀 then read on 👇

### 🤗 Hosting Your Telegram Bot
One little known feature 🤫 that I discovered recently is that you can host your Telegram bot on Hugging Face Spaces. 

If you create a **new** Space and upload the `app.py`, it will work out of the box! 
Now you don't have to keep you computer alive 24/7 to run the bot.

I'm not sure if this feature is intentional or not by Hugging Face, but this is pretty neat eh? Free hosting for your bots! 😎

To make sure we don't expose our Telegram **token** in the source code, let's set the token to be an environment variable.

On your Space, click on the `Settings` tab and enter the `Name` and `Value` of the environment variable.
Let's put the name as `telegram_token` and the value is your Telegram **token**.
{{< figure_resizing src="secrets.png" >}}

On your app.py change `line 31` to the following

```python
updater = Updater(os.environ['telegram_token'])
```

Now, you can freely share your codes without exposing your Telegram token!

{{< notice tip >}}
For completeness, you can view my final `app.py` [here](https://huggingface.co/spaces/dnth/ptb-gpt/blob/main/app.py).
{{< /notice >}}

<!-- https://huggingface.co/spaces/dnth/ptb-gpt -->


<!-- `Line 31` loads the token you've set as environment variable.
`Line 32` detects when the user sends the `/start` command and calls the `hello` function.
`Line 33` detects texts that are non-commands and calls the `respond_to_user` function. -->

### 🎉 Conclusion

Link to Telegram bot
https://t.me/ptbgptbot


### 🙏 Comments & Feedback
I hope you've learned a thing or two from this blog post.
If you have any questions, comments, or feedback, please leave them on the following Twitter post or [drop me a message](https://dicksonneoh.com/contact/).
{{< tweet dicksonneoh7 1523250980233510912>}}


If you like what you see and don't want to miss any of my future contents, follow me on Twitter and LinkedIn where I deliver more of these tips in bite-size posts.