# Covid-19 Tweets Sentiment Analysis

# Dataset Source

The dataset has been obtained from kaggle. Link-https://www.kaggle.com/datatattle/covid-19-nlp-text-classification

# Data Dictionary

The dataset contains following features:

1) Username - User ID
2) Screen Name - Tweet ID
3) Location - From where the tweet was posted.
4) Tweet At - The time at which the tweet was posted.
5) Original Tweet - Content of tweets
6) Sentiment - Containing sentiment of the tweets(Positive,Extremely Positive,Negative,Extremely negative and Neutral).

# Codes and Resources Used

Python version - 3.7.10 (in runtime.txt)

Packages: pandas, numpy, re, scikit-learn, tensorflow, matplotlib, NLTK, Scrapy, Textblob, seaborn, plotly, pickle, and streamlit.
 
For Web Framework Requirements: pip install -r requirements.txt

Cloud Platform for deployment - Heroku.

# Data Cleaning and EDA

After loading of data, I first examined which features are relevant to the data and removed unnecessary features.Categorized sentiments into positive,neutral and negative values and found out the distribution of them in both training and test dataset.

For text cleaning, removed unwanted punctuations and symbols using regular expressions,removal of short words and converted all elements in lower case.

For EDA,by use of histograms found out basic info such as number of words in tweets,number of characters in tweets and average word length.

Analyzed most frequent stopwords,most frequent words,sentiment polarity of tweets,top unigrams,bigrams and trigrams in both train and test sets.

Identified the part of speech of words using pos tagging in textblob.

Identified most important words using wordcloud in all sentiments of tweets and lastly performed lemmatization using spacy.

# Model Building

Hard Encoding of sentiments.

Calculating the maximum length of sentences.

Performing tokenization using keras tokenizer with padding of sequences to maximum length of sentences calculated above.

For better generalization of context,using 100-dimensional glove word embeddings and generating embedding matrix according to our requirements.

Using keras embedding layer,Bidirectional LSTMs with dropout layers and dense layers to generate predicitons.

Evaluating the model using accuracy as dataset is balanced.

Generated predicitons were 82% accurate in test set.

Saving the model as hdf5 file for model deployment.

# Productionization

Loading the saved model.

Performing text pre-processing and vectorization using tokenizer used in training for new inputs.

The model was converted into an API using streamlit and deployed in Heroku.

The API takes tweets as an input through an form built using streamlit and then make predictions in form of the sentiment and its score.

API Link=https://covid-tweet-sentiment-analysis.herokuapp.com/
