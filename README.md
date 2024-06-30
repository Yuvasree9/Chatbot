
# Chatbot

This repository contains a Google colab notebook Notebook demonstrating the implementation of a simple chatbot using Python and the NLTK library.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Code Explanation](#code-explanation)
  - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
    - [Import Libraries](#import-libraries)
    - [Load Data](#load-data)
    - [Text Preprocessing](#text-preprocessing)
  - [Model Training](#model-training)
    - [Bag of Words Model](#bag-of-words-model)
    - [Training the Model](#training-the-model)
  - [Chatbot Interaction](#chatbot-interaction)
    - [User Input](#user-input)
    - [Generating Response](#generating-response)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project demonstrates how to build a simple rule-based chatbot using natural language processing techniques. The chatbot can understand and respond to user inputs based on predefined rules and patterns.

## Prerequisites
- Python 3.x
- Google Colab Notebook
- NLTK library

## Installation
 Clone the repository:
   ```bash
   git clone https://github.com/Yuvasree9/Chatbot.git
```

<h2>Usage</h2>
<ol><li>Open the Jupyter Notebook or Google colab notebook.</li><li>Execute the cells in the notebook sequentially to train the chatbot and interact with it.</li></ol>
<h2>Features</h2>
<ul><li>Text processing and tokenization</li>
<li>Training a chatbot model using a predefined dataset</li>
<li>Interacting with the chatbot through text input</li></ul>
<h1>Code Explanation</h1>
<h3>Data Loading and Preprocessing</h3>
<h5>Import Libraries</h5>
<ul><li>Import necessary libraries such as NLTK for natural language processing, numpy for numerical operations, and string for string operations.</li></ul>

```
import nltk
import numpy as np
import random
import string
Load Data
```

<h5>Load Data</h5>
<ul><li>Load the dataset which contains the data the chatbot will be trained on.
</li></ul>
<h5>Preprocessing Data</h5>
<ul><li>Preprocess the loaded data by tokenizing, lemmatizing, and removing punctuation.</li></ul>
<h5>Text Preprocessing</h5>
<ul><li>Preprocess the loaded data by tokenizing, lemmatizing, and removing punctuation.</li></ul>

```
nltk.download('punkt')
nltk.download('wordnet')
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)
```

<h3>Model Training</h3>
<h5>Bag of Words model</h5>
<ul><li>Create a Bag of Words model to convert text data into numerical data that the machine learning algorithm can understand.</li></ul>

```
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

<h5>Training the model</h5>
<ul><li>Train the chatbot model using the processed dataset and the Bag of Words model.</li></ul>

```
def response(user_response):
    robo_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response
```
<h3>Chatbot Interaction</h3>
<h5>User Input</h5>
<ul><li>The user can input text, which the chatbot will process and generate a response to.</li></ul>

```
flag = True
print("BOT: My name is Chatbot. I will answer your queries. If you want to exit, type Bye!")
while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print("BOT: You are welcome..")
        else:
            if(greeting(user_response) != None):
                print("BOT: " + greeting(user_response))
            else:
                print("BOT: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("BOT: Bye! take care..")
```

<h5>Generating Response</h5>
<ul><li>Based on the user input, the chatbot generates an appropriate response using cosine similarity to find the most relevant answer from the dataset.</li></ul>

<h2>Contributing</h2>
<p>Contributions are welcome! Please feel free to open an issue or submit a pull request with any improvements.</p>

   
