import itertools
import pytest
import numpy as np
import SentimentAnalysis

"""Author: Ethan Krafft
   Course: CSC 439 NLP
   Instructor: Dr. Peter Jansen
   
   Description: SentimentTest.py is used to for training, development, and 
       testing of the 

"""

def test_read_imdb_reviews():
    train_count = 0
    for sentence in SentimentAnalysis.read_review_data("imdb_train.txt"):
        assert len(sentence) == 2
        train_count += 1
    assert train_count > 595 and train_count < 605
    
   
    
    test_count = 0
    for sentence in SentimentAnalysis.read_review_data("imdb_test.txt"):
        assert len(sentence) == 2
        test_count += 1
    assert test_count > 195 and test_count < 205
    
    dev_count = 0
    for sentence in SentimentAnalysis.read_review_data("imdb_dev.txt"):
        assert len(sentence) == 2
        dev_count += 1
    assert dev_count > 195 and dev_count < 205
    
test_read_imdb_reviews()


def test_create_BOW():
    corpus = SentimentAnalysis.read_review_data("imdb_train.txt")

    bag_of_words, labels = SentimentAnalysis.create_BOW(corpus)
    assert(len(bag_of_words) == 2)
    assert(0 in bag_of_words)
    assert(1 in bag_of_words)
    
    
    corpus2 = SentimentAnalysis.read_review_data("imdb_train.txt")
    features = SentimentAnalysis.create_Features(corpus2)
    assert len(features) > 2300 and len(features) < 2400

    
test_create_BOW()
    



def test_train_dev_naive():
    corpus = SentimentAnalysis.read_review_data("imdb_train.txt")
    bag_of_words, labels = SentimentAnalysis.create_BOW(corpus)
    corpus2 = SentimentAnalysis.read_review_data("imdb_train.txt")
    features = SentimentAnalysis.create_Features(corpus2)
    
    
    baseline_model = SentimentAnalysis.naive_baseline(bag_of_words, labels, features)
    assert(len(baseline_model) > 1500 and len(baseline_model) < 2400)
    
    experimental_model = SentimentAnalysis.naive_experimental(bag_of_words, labels, features)
    assert(len(baseline_model) > 1500 and len(baseline_model) < 2400)
    
    print("BASELINE TESTING RESULTS")
    print("-------------------------------------------------------------------")
    temp_corpus = SentimentAnalysis.read_review_data("imdb_dev.txt")
    dev_bow, dev_labels = SentimentAnalysis.create_BOW(temp_corpus)
    dev_corpus = SentimentAnalysis.read_review_data("imdb_dev.txt")
    SentimentAnalysis.test_model(baseline_model, dev_corpus)
    print("\n\n")
    print("EXPERIMENTAL TESTING RESULTS")
    print("-------------------------------------------------------------------")
    temp_corpus2 = SentimentAnalysis.read_review_data("imdb_dev.txt")
    dev_bow2, dev_labels2 = SentimentAnalysis.create_BOW(temp_corpus2)
    dev_corpus2 = SentimentAnalysis.read_review_data("imdb_dev.txt")
    SentimentAnalysis.test_model(experimental_model, dev_corpus2)
    
    

test_train_dev_naive()
    
    




























