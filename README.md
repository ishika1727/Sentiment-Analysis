# Sentiment-Analysis

## Introduction

In this project, we address the problem of sentiment analysis on a corpus of text data. We use a number of machine learning and deep learning methods to perform sentiment analysis. Following which we will choose the best fit model based on their accuracy scores and use that for our sentiment analysis. The experimental results suggest that using a particular method is subjective to the application.

## DATASET DESCRIPTION & SAMPLE DATA
The data given is in the form of comma-separated values files with tweets and their
corresponding sentiments. The training dataset is a csv file of type tweet_id, sentiment, tweet
where the tweet_id is a unique integer identifying the tweet, sentiment is either 1 (positive) or
0 (negative), and tweet is the tweet enclosed in "". Similarly, the test dataset is a csv file of
type tweet_id, tweet.
The dataset is a mixture of words, emoticons, symbols, URLs and references to people.
Words and emoticons contribute to predicting the sentiment, but URLs and references to
people don’t. Therefore, URLs and references can be ignored. The words are also a mixture
of misspelled words, extra punctuations, and words with many repeated letters. The tweets,
therefore, have to be preprocessed to standardize the dataset

![image](https://user-images.githubusercontent.com/72199738/135661532-ea0c08fe-8671-4bce-888b-41102105a68f.png)
