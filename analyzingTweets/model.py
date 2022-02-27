import numpy as np
import pandas as pd
import vaderSentiment
import csv
from sklearn.model_selection import train_test_split
import model_helper as mh
df = pd.read_csv('ukraine_war_dataset_2_24-25_2022.csv')

def run_model():
    pos_count = 0
    neg_count = 0
    neu_count = 0

    df_copy = df.copy()
    print("normalizing data...")
    df_copy['text'] = mh.normalization(df_copy)

    print("loading label...")
    labels = mh.sentiment_analysis(df_copy['text'])
    custom_neg, custom_neg_specific = mh.error_modificators()

    print("modifying error...")
    mh.modify_error(df_copy, labels, custom_neg)

    print("Total Tweets: ", len(df_copy))

    pos_count = labels.count(1)
    neg_count = labels.count(-1)
    neu_count = labels.count(0)
    print("Positive: ", pos_count / len(df_copy), pos_count)
    print("Negative: ", neg_count / len(df_copy), neg_count)
    print("Neutral: ", neu_count / len(df_copy), neu_count)

    labels, pos_neg_data, pos_neg_labels = mh.get_pos_neg(df_copy, labels)

    processed_features, vectorizer = mh.vectorization(pos_neg_data)

    X = processed_features
    y = pos_neg_labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    text_classifier, train_accuracy = mh.train_model(X_train, y_train)
    test_accuracy = mh.test_model(text_classifier, X_test, y_test)

    print(train_accuracy, test_accuracy)
    return text_classifier, vectorizer, custom_neg_specific, labels

def predict(user_input, text_classifier, vectorizer, custom_neg_specific):
    tmp = []
    tmp.append(user_input)

    prediction = text_classifier.predict(vectorizer.transform(tmp))[0]
    if prediction == 1 and any(w in user_input for w in custom_neg_specific):
        prediction = -1
    return prediction

def recmd_tweets(prediction, labels):
    if prediction == 1:
        rcmd_tweets_idx = np.where(labels == 1)
    else:
        rcmd_tweets_idx = np.where(labels == -1)
    rcmd_tweets = df.iloc[rcmd_tweets_idx]
    rcmd_top_likes = rcmd_tweets.nlargest(5, 'like_count')
    top_likes = df.nlargest(5, 'like_count')
    return rcmd_top_likes, top_likes
