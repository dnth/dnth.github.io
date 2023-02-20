---
title: "ML Pipelines from the Get Go (Without Tears)"
date: 2022-08-25T20:48:15+08:00
featureImage: images/blog/talk_ml_pipelines_from_the_get_go/thumbnail.gif
postImage: images/blog/talk_ml_pipelines_from_the_get_go/post_image.jpeg
tags: ["open-source", "deployment"]
categories: ["invited-talks"]
toc: true
socialshare: true
description: "Deploy and share your models to iterate quickly."
images : 
- images/blog/talk_ml_pipelines_from_the_get_go/post_image.jpeg
---

### 💫 Takeaways 
In this talk, I discussed why ML pipelines should be built from the get-go. Here are some of the key takeaways:

1️⃣ Despite the hype around machine learning, 55% of companies have not deployed a single ML model yet, and those who have are struggling to maintain and scale them.

2️⃣ Putting ML in production is not just about ML, but also about engineering. Many companies are doing more engineering than ML to solve various issues that arise in the pipeline.

3️⃣ There are many ML Ops tools and platforms available, but this can be overwhelming and disorienting for beginners. It's important to start with a clear plan and a focus on the most critical parts of the pipeline.

4️⃣ Data cleaning and preprocessing can take up to 80% of the data scientist's time, so it's important to have efficient processes and tools to handle this.

Watch the full video to learn more about these challenges and how to tackle them! 🎥👨‍🏫

The talk was hosted with FuzzyLabs and Skillsmatter.
View the talk on Skillsmatter webpage [here](https://skillsmatter.com/skillscasts/17873-an-introduction-to-zenml).

<iframe src="https://www.youtube-nocookie.com/embed/RlUfYCIPUNI" title="YouTube video player" onload='javascript:(function(o){o.style.height=o.contentWindow.document.body.scrollHeight+"px";}(this));' style="height:500px;width:100%;border:none;overflow:hidden;" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

### 🔥 Motivation

Machine Learning (ML) has become an integral part of various sectors, including finance, healthcare, e-commerce, and more. Despite its popularity, many companies struggle to deploy and maintain ML models in production, with 55% of companies yet to deploy a single ML model.

ML pipelines offer a solution to this problem. They're a sequence of steps that enable organizations to manage, deploy, and maintain their ML models in a production environment. 

In this talk, we discuss why ML pipelines should be built from the get-go and how they can help organizations overcome the challenges of deploying and maintaining their ML models. We will also explore the different stages of an ML pipeline, the benefits of using ML pipelines, and best practices for building them. So, let's get started!

### ✅ Why ML Pipelines

The process of implementing machine learning models can be complex and challenging. While developing a model is an important aspect, it's just one piece of the puzzle. There are several other steps involved, such as data cleaning, feature engineering, training, testing, deployment, and monitoring. Therefore, to ensure a smooth and efficient process, it's essential to build a machine learning pipeline from the get-go.

A machine learning pipeline is a set of processes that help in managing and automating machine learning workflows. It involves several stages, such as data collection, data cleaning, data pre-processing, feature extraction, model training, model evaluation, model deployment, and monitoring. Building a pipeline allows you to automate most of these processes, saving you time and effort in the long run.

Moreover, an ML pipeline helps in maintaining consistency, accuracy, and reproducibility of results. With a well-built pipeline, you can ensure that every time you run the model, you'll get the same results. This is crucial, especially in the industry, where minor discrepancies can result in significant losses.

### 🤖 Challenges of Implementing ML Pipelines

Despite the growing popularity of machine learning, many companies have yet to deploy any machine learning models, and even those who have deployed them are struggling to maintain them. In fact, according to recent studies, more than 55% of companies have not deployed a single machine learning model yet. For those that have, their models are not deployed to a majority of their products, but instead comprise only a small portion of their products. This raises the question of what went wrong.

The dream of machine learning in production is that data collection, training the model, and putting it into production are three steps that happen in sequence. However, the reality is that the process is not that straightforward. In actuality, it looks more like an infinite cycle of back and forth between data collection, training the model, testing, and deployment. This cycle can take a long time, and some people continue looping until their model is good enough to put into production.

The majority of the problem with putting ML in production is not in the ML itself, but it's more of the engineering side of things. Most of the time, people are doing more engineering than ML. To solve various engineering issues, it takes a lot of time, and there are now a lot of companies that are jumping in to help with all of these issues. This has resulted in hundreds of tools, and if you're new to ML Ops, it can be overwhelming, and you don't know where to start or which one to pick.

The problem with inefficiency is also a challenge in ML in the industry. According to a survey conducted in 2012, the 80/20 data science dilemma reveals that most data scientists spend most of their time finding, cleaning, and reorganizing huge amounts of data, which is an inefficient strategy when the real thing they should be focusing on is building the model.

### 🎄 How to Implement an ML Pipeline

There are many existing tools available to implement machine learning pipelines. Some of the most popular ones are [Apache Airflow](https://airflow.apache.org/), [Kubeflow](https://www.kubeflow.org/), [TensorFlow Extended](https://www.tensorflow.org/tfx), and [ZenML](https://zenml.io/home).

An advantage of ZenML is that its an open-source MLOps framework to unify all the components in your pipeline. With ZenML you can streamline your ML workflows, improve collaboration, and accelerate your results. Check out the GitHub repo [here](https://github.com/zenml-io/zenml).

![img](stack.gif)

Here is a full list of all stack components currently supported in ZenML, 
with a description of that components role in the MLOps process:


{{<table "table table-striped table-bordered">}}


| **Component**                                          | **Description**                                                   |
|----------------------------------------------------------------------|-------------------------------------------------------------------|
| Orchestrator                     | Orchestrating the runs of your pipeline                           |
| Artifact Store               | Storage for the artifacts created by your pipelines               |
| Container Registry | Store for your containers                                         |
| Secrets Manager           | Centralized location for the storage of your secrets              |
| Step Operator                  | Execution of individual steps in specialized runtime environments |
| Model Deployer              | Services/platforms responsible for online model serving           |
| Feature Store                  | Management of your data/features                                  |
| Experiment Tracker   | Tracking your ML experiments                                      |
| Alerter                                  | Sending alerts through specified channels                         |
| Annotator                              | Labeling and annotating data                                      |
| Data Validator              | Data and model validation                                         |
| Image Builder                 | Builds container images.                                          |
{{</table>}}




Regardless of which tool you pick, it is important to make sure you are clear with each step that takes place in the pipeline. Here are some of the important points mentioned in the talk:

+ **Define your objectives**: Before you start building your ML pipelines, it's essential to define your objectives. What are you trying to achieve with your model? What kind of data will you need? How will you evaluate the performance of your model?

+ **Data collection**: Once you have defined your objectives, you need to collect the data that you will use to train your model. It's crucial to ensure that your data is clean, accurate, and representative of the problem you're trying to solve.

+ **Preprocessing**: Once you have your data, you need to preprocess it to prepare it for your model. This may involve cleaning, normalization, feature engineering, and more. It's important to ensure that your preprocessing steps are well-documented and reproducible.

+ **Model training**: After preprocessing, you can train your model using your data. There are many different algorithms and techniques you can use, depending on your objectives and the nature of your data. Again, it's essential to keep track of your experiments and document your results.

+ **Model evaluation**: Once you have trained your model, you need to evaluate its performance. This may involve testing your model on a holdout dataset or using cross-validation techniques. It's important to use appropriate evaluation metrics and to be aware of any potential biases in your data.

+ **Deployment**: Once you have a model that performs well, you need to deploy it in a production environment. This may involve integrating your model with other systems, creating APIs, and ensuring scalability and reliability. It's important to test your model thoroughly before deployment and to monitor its performance in production.

By following these steps, you can build ML pipelines from the get-go that are robust, efficient, and effective.


In addition to the points mentioned above, it's important to emphasize the need for model monitoring. Once a model is deployed into production, it's crucial to keep an eye on its performance and make sure it's still providing accurate predictions. This is where model monitoring comes in.

By monitoring the model's performance metrics, such as accuracy and precision, we can quickly detect if the model is not performing as expected. This can be done through various tools, such as dashboards and alerts, that notify the team if the model's performance drops below a certain threshold.

Moreover, model monitoring allows for continuous improvement of the model. By analyzing the performance data and identifying areas for improvement, the team can retrain the model with new data to improve its accuracy and reliability.

### 🌿 Conclusion
Machine learning in production goes beyond training models. Having an ML pipeline from the get go ensures reproducibility and reliability of the model in production.

There are various MLOps tools to build a pipeline. Pick the one you're comfortable with considering various aspects such as interoperability, ease of use and collaboration support. 