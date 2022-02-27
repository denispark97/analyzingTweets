import numpy as np
import pandas as pd
import vaderSentiment
import csv
import nltk
from nltk.corpus import stopwords
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

def normalization(df):
    texts = df['text']
    texts = texts.str.lower()
    texts.apply(lambda x: re.sub("[^a-z\s]", "", x))
    texts = texts.apply(lambda x: " ".join(word for word in x.split() if word not in set(stopwords.words("english"))))
    return texts

def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    
    if sentiment_dict['compound'] >= 0.05:
        return "Positive"
    elif sentiment_dict['compound'] <= - 0.05:
        return "Negative"
    else:
        return "Neutral"

def sentiment_analysis(texts):
    num_pos = 0
    num_neg = 0
    num_neu = 0
    labels = []
    for i in range(len(texts)):
        sentiment = sentiment_scores(texts[i])
        if sentiment == "Positive":
            labels.append(1)
        elif sentiment == "Negative":
            labels.append(-1)
        else:
            labels.append(0)
    return labels

def error_modificators():
    custom_neg = ['support ukraine','love ukraine','pray','praying','heart','god bless ukraine','prayer','horrible','stay safe','peace',
              'fuck putin','stop war','disaster','aggression','help ukraine', 'god bless #ukraine', 'god help', 'stay strong', 'shame', 'victim',
              'mercy']

    custom_neg_specific = ['support ukraine','love ukraine','pray for ukraine', 'heart','god bless ukraine','prayer','horrible','stay safe','peace',
              'fuck putin','stop war','disaster','aggression','help ukraine', 'god bless ukraine', 'god help', 'stay strong', 'victim']
    return custom_neg, custom_neg_specific

def modify_error(df_copy, labels, custom_neg):
    for i, text in enumerate(df_copy['text']):
        if labels[i] == 1 and any(x in text for x in custom_neg):
            labels[i] = -1

def get_pos_neg(df, labels):
    nplabels = np.array(labels)
    pos_idx = np.where(nplabels == 1)[0]
    neg_idx = np.where(nplabels == -1)[0]
    pos_neg_idx = pos_idx.tolist() + neg_idx.tolist()
    pos_neg_data = df.iloc[pos_neg_idx]
    pos_neg_labels = nplabels[pos_neg_idx]
    return nplabels, pos_neg_data, pos_neg_labels

def vectorization(pos_neg_data):
    vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
    processed_features = vectorizer.fit_transform(pos_neg_data['text']).toarray()
    return processed_features, vectorizer

def train_model(X_train, y_train):
    text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    text_classifier.fit(X_train, y_train)
    return text_classifier, text_classifier.score(X_train, y_train)

def test_model(text_classifier, X_test, y_test):
    y_pred = text_classifier.predict(X_test)
    return text_classifier.score(X_test, y_test)