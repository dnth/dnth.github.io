---
title: "Contributing to open-source: Lessons learned"
date: 2022-04-04T20:48:15+08:00
featureImage: images/blog/contributing_to_open_source_lessons_learned/feature_image.gif
postImage: images/blog/contributing_to_open_source_lessons_learned/post_image.jpeg
tags: ["open-source", "IceVision", "lessons-learned"]
categories: ["self-development"]
toc: true
socialshare: true
description: "First Pull Request to a core developer of IceVision in 3 months."
images : 
- images/blog/contributing_to_open_source_lessons_learned/post_image.jpeg
---

### ☀️ Introduction
I was recently given the recognition by the folks at [IceVision](https://github.com/airctic/icevision) to be part of the core-developer team!
[Farid Hassainia](https://www.linkedin.com/in/farid-hassainia-ca/), a co-creator of the IceVision Tweeted about it on March 17, 2022. 
{{< tweet 1504457803552935943>}}

### 🏅 Lessons learned
Reflecting on how far I've come, I am astonished and humbled.
Just 3 months ago, I learned how to open my first *Pull Request* on GitHub. 

Despite having coded for over 10 years, I never bothered to learn about something so commonly used in the software world - *Git*, a version control system.
I took the plunge, and the first step was painful. 
But every subsequent step got better and better. 

I never looked back since.

Even though it's not late, I wish I had started earlier. 
If you are new and want to contribute to open-source, start now.
Here are some of the lessons I learned along the way that changed the way I learn, and code.

{{< figure_resizing src="tree_proverb.jpg" >}}


#### 🧑‍🔧 I became a better coder
I've spent years coding and dabbling in many projects since I was a student.
With so much time spent on coding, one would be expected to have a good grasp of it.
But in reality, I didn't know how good or bad my codes were. 
Or whether the codes are of best practices or optimal in any way given that a significant portion of the codes was taken off snippets from various sources like Stackoverflow, blog posts, and even GitHub.

Most of the time, when the code works, I leave it as it is, and when it doesn't, I try to troubleshoot to the best of my ability.
I can recall on numerous occasions when my codes didn't work out, and I didn't know why. 
And sometimes the codes worked, and I still have no idea why 🤦‍♂️.

Because the codes were not publicly shared, there was very limited feedback or insights that anyone can offer to improve them even when I asked for help.
As a result, bugs in the codes remained as bugs, and they never got fixed.
This severely limited what I could have learned and gained had I gotten better feedback by sharing my codes publicly.
This is the major difference between learning in private and in public.

{{< tweet 1504850040963092480>}}

Learning in public is many times more powerful compared to learning in private.
Nowadays, it's so easy to start learning in public by contributing to open-source software.
All you have to do is pick the one you are most interested in and start engaging the community around the project.

For me, the project that got me interested was [IceVision](https://airctic.com/0.12.0/).
To those unfamiliar, IceVision is a computer vision library built on top of [Fastai](https://github.com/fastai/fastai) to make applying deep learning to images very easy for new users. 
Without much coding experience, you can get started and train your own object detection model in a few lines of code within minutes.
Checkout their tutorial [here](https://github.com/airctic/icevision/blob/master/notebooks/getting_started_object_detection.ipynb).

At first, all I did was try to use the library for my own side projects.
Gradually, I realized there are limitations to the existing features of the library. I started asking for help from the community on how can I overcome the limitations by proposing a new feature.
This was how I discovered that there are ways I can contribute to the library by adding the features I needed.

With IceVision, I have always wanted to use [Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) model for object detection tasks. 
However, that was not included in the library at that point.
I then started to research and ask for help on how to include that model in the library.
That was the beginning of how I started contributing to the library.
I ended up adding more than 10 models to the library in 14 pull requests. 
A dozen commits and thousands of lines of codes later, I became an active contributor to the library. 
The rest is history.

Little that I know, I was ranked #26 on [Top GitHub Users By Country](https://github.com/gayanvoice/top-github-users/blob/main/markdown/public_contributions/malaysia.md) out of 1000+ contributors from Malaysia. 
The ranking is based on the number of public contributions on GitHub.
All these happened within a short span of 3 months!
{{< figure_resizing src="top_cont.png" >}}

#### 👨‍🏫 I got mentorship from coding experts
Getting a mentor is no doubt one of the best investments for rapid progress in any area of personal growth.
Ask any successful figures for advice, and getting mentors will surely top the list.
In my journey of contributing to open-source, one of the most rewarding experiences is the opportunity to learn from the maintainers of the library as they help and guide me to contribute to the library.
More often than not, these maintainers (especially for reputable libraries) are highly experienced coders.

On one occasion, I got the opportunity to get on a video call with Farid to discuss how to fix an issue with the newly added features of IceVision.
Even though our interactions were purely online, I got a glimpse into his thought process as he explained how things are done in IceVision.
Needless to say, I learned a lot, not only on the technical aspect but also on the elusive non-technical side such as the mindset of a developer.
Farid was also kind enough to offer me practical advice on how to stand out as a developer and encouraged me to blog about what I learned.
This blog exists thanks to his advice and encouragement.

As I familiarize myself more with the library, I also have had many opportunities to help others who are struggling with the same problems I had.
In the process of helping others solve their problems, I often find myself learning new things and exploring areas I never bother to look into.
As I help others, I gained and learned a lot myself which allowed me to contribute more to the library.
This became an empowering cycle to continue contributing to the library.
One day, I hope to be a mentor of some sort to others and help inspire people in the ways I was inspired.

{{< figure_resizing src="mentor.png" >}}


#### 🌎 I started a website from scratch
Encouragement from Farid led me to start blogging.
In the process, I explored and learned how to use several blogging platforms including Wordpress, Ghost, and Hugo.

I finally settled with [Hugo](https://gohugo.io/) as it gives the best flexibility for my use.
In the process, I also learned about self-hosting a webpage on GitHub pages and Digital Ocean, purchasing a domain, and GitHub workflows. 

In building this website, I found myself contributing to the Hugo [theme](https://github.com/StaticMania/portio-hugo) that I used for this site. I learned how Hugo themes work and a little about [Go language](https://go.dev/) used in the theme.
By this time, I became very comfortable with *Pull Requests* even though it was only weeks after I stumbled upon Hugo.

The learning curve was steep, but it's well worth it.

Additionally, in the process of blogging, I also realized that writing about something really helped me solidify and clarify my understanding of the things I write about.
More often than not, I find myself in the illusion of understanding something when I really don't.
Strangely, this is only evident when I start writing about it.

I guess this is because writing is an act of synthesizing from a knowledge pool which can only be accomplished when one has a good understanding of the concept.
This is why one of the greatest physicists of all time Richard Feynman once said "If you want to master something, teach it".

{{< figure_resizing src="explain_quote.jpeg" >}}

Learning by writing is also a form of active learning as opposed to passively absorbing content.
This has significantly improved my understanding of many concepts I thought I already knew.

Apart from that, I also shared my blog posts on Twitter and LinkedIn which resulted in the online presence and visibility of my blog.
Once again the IceVision community is very kind to share my posts and provided me some valuable feedback.

Starting a website and learning its intricacies are something that I would never venture into had I not contributed to open-source.
I never expected this, and I find this a huge bonus learning just because I started contributing to IceVision.


#### 🙌 I learned about the kindness of strangers
Through my experience in contributing to open-source, I learned that there are lots of kind people out there who are more than willing to help.
You will just have to find them.

In the IceVision community, there are many such people, and I am indebted to them for helping me in becoming a better coder and a better person. 
I got to know people from around the globe from various backgrounds and cultures.
This also helped me build my self-esteem as I interact with them.
I also learned to appreciate that we all are gifted in different ways and everyone has unique combinations of talents and skills - just like members of the Avengers.

In a short period of time, I made a few friends and expanded my social circle.
Through them, I also managed to score a few job application interviews, as I was looking for new employment opportunities.

P/S: I am still looking for an employment opportunity, I would be forever grateful if you can [connect me](https://dicksonneoh.com/contact) to anyone who's hiring for computer vision or machine learning positions.

{{< figure_resizing src="friend_quote.jpeg" >}}

### 🎁 Wrapping up
It's a wrap! 
Contributing to open-source has undoubtedly changed my perspective quite a bit in merely 3 months.

I have learned valuable lessons in becoming a better coder, a writer, and a better person.
I wish to also thank the folks at IceVision for making such an awesome deep learning computer vision library available.

Special shoutout to [Farid](https://www.linkedin.com/in/farid-hassainia-ca/) and [Francesco](https://www.linkedin.com/in/francescopochetti/) for helping me out along the way.
If you're interested, do join in the [Discord](https://t.co/CDIWhdVmSe) channel or get in touch with me on [Twitter](https://twitter.com/dicksonneoh7) or [LinkedIn](https://www.linkedin.com/in/dickson-neoh-3a6984b8/).


{{< figure_resizing src="ice2.png" caption="Photo from our most recent meetup - IceVision core developers from around the globe.">}}

### ⛏ Comments & Feedback
If you have any questions, comments, or feedback, I would be grateful if you can leave them on the following Twitter post or [drop me a message](https://dicksonneoh.com/contact/).
{{< tweet 1511269785010548739>}}