from typing import List, Text
import spacy
nlp = spacy.load('en_core_web_sm')

"""Author: Ethan Krafft
   Course: CSC 439 NLP
   Instructor: Dr. Peter Jansen
   
   Description: SentimentAnalysis.py is the main program for the CSC 439 final
       project. It includes a baseline model that utilizes a naive-bayes 
       classifier and an experimental model that seeks to improve on the baseline
       through devlopment and testing datasets. It is run through the partner
       program SentimentTest.py.

"""
def read_review_data(review_data: str):
    """This function is used to process the initial data including strings of
        sentence reviews for a movie, and a integer rating.
    """
    file_list = open(review_data).read().splitlines()
    split_list, token_list, sentence_list = [],[],[]
    for line in file_list:
        split = line.strip().split('\t')
        if len(split) == 2:
            split_list.append(split)
    for sent in split_list:
        token_list = tokenize_text(sent[0].strip())
        sentence_list.append((token_list, sent[1]))
        token_list = []
    sentence_iter = iter(sentence_list)
    return sentence_iter



def tokenize_text(text: Text) -> List[Text]:
    """This function tokenize a text by iterating 
            over its tokens by using spacy English tokenizer.
        =============
        Params:
            text
        Return:
            A list of tokenized items. ["token1, "token2"...etc]
    """
    text_doc = nlp(text)
    return_list = []
    templist = [[token.text for token in sent] for sent in text_doc.sents]
    for x in templist:
        for y in x:
            return_list.append(y)
    return return_list



def create_BOW(corpus: iter):
    """This function is responsible for creating a bag-of-words used in the
        naive bayes classifier later in the project. It returns a dictionary
        form of a bag-of-words and a count of labels showing the number of 
        positive or negative reviews in a train/dev/test set. 
    """
    bow_pos, bow_neg, label_count = {},{},{}
    label_count[0] = 0
    label_count[1] = 0
    bag_of_words = {}
    for sentence in corpus:
        review = int(sentence[1])
        label_count[review] += 1
        for token in sentence[0]:
            if review == 0:
                if token not in bow_neg:
                    bow_neg[token] = 1
                else:
                    bow_neg[token] += 1
            elif review == 1:
                if token not in bow_pos:
                    bow_pos[token] = 1
                else:
                    bow_pos[token] += 1
 
    bag_of_words[0] = bow_neg
    bag_of_words[1] = bow_pos
    return bag_of_words, label_count



def create_Features(corpus: iter):
    """Method is used to create a set of features indicating which 
        tokens are in the documents and how often a feature is associated with 
        a positive or negative rating. 
    """
    features = {}
    for sentence in corpus:
        review = int(sentence[1])
        for token in sentence[0]:
            if token not in features:
                features[token] = {}
                features[token][0] = 1
                features[token][1] = 1
            else:
                features[token][review] += 1
 
    return features



def naive_baseline(bag_of_words, labels, features):
    """Function is used to create a baseline model of a naive-bayes classifier
        given a bag-of-words, label counts, and feature set. 
    """
    word_probability = {}
    for word in features:
        if len(word) > 4:
            p_token_given_pos = (features[word][1])/labels[1]
            p_pos = (features[word][1])/(features[word][0] + features[word][1])
            p_token = (features[word][0] + features[word][1])/(labels[0] + labels[1]) 
            final_prob = (p_token_given_pos * p_pos)
            final_prob = final_prob/(p_token)
            word_probability[word] = final_prob
    return word_probability 

def naive_experimental(bag_of_words, labels, features):
    """Function is used to create an experimental model of a naive-bayes 
        classifier given a bag-of-words, label counts, and feature set.
    """
    word_probability = {}
    for word in features:
        p_token_given_pos = (features[word][1])/labels[1]
        p_pos = (features[word][1])/(features[word][0] + features[word][1])
        p_token = (features[word][0] + features[word][1])/(labels[0] + labels[1]) 
        final_prob = (p_token_given_pos * p_pos)
        final_prob = final_prob/(p_token)
        word_probability[word] = final_prob
    return word_probability
        

def test_print(actual_positive, actual_negative, correct_positive, correct_negative,
               incorrect_positive, incorrect_negative, result_positive, result_negative):
    """Function is used to print the testing results from the baseline and 
        experimental models. It prints some accuracy values, but more importantly
        it shows precision for positive and negative classifications.
    """
    precision = correct_positive/(correct_positive + incorrect_positive)
    recall = correct_positive/(correct_positive + incorrect_negative)
    f1_score = 2*(precision*recall)/(precision + recall)
    print("correct positive: " + str(correct_positive))
    print("correct negative: " + str(correct_negative))
    print("incorrect positive: " + str(incorrect_positive))
    print("incorrect negative: " + str(incorrect_negative))
    
    print(" ")
    print("actual positive expected: " + str(actual_positive))
    print("result positive: " + str(result_positive))
    print(" ")
    
    print("positive classification precision: " + str(correct_positive/result_positive))
    print("negative classification precision: " + str(correct_negative/result_negative))
    
    difference = abs(actual_positive - result_positive)
    total_accuracy = (actual_positive - difference)/actual_positive
    print(' ')
    print("Model f1 score: " + str(f1_score))
    print("Model total accuracy: " + str(total_accuracy))
    print("\n")
    
    
    
def test_model(model, corpus):
    """Function is used to gather testing data over both the baseline and 
        experimental models. It counts classification results for positive and 
        negative classifications, as well as correct/incorrect classifications. 
    """
    actual_positive,actual_negative = 0,0
    correct_positive, correct_negative  = 0,0
    incorrect_positive, incorrect_negative = 0,0
    result_positive, result_negative = 0,0
    word_probs = []
    for sent in corpus:
        sent_prob = []
        if sent[0] == '0':
            actual_negative += 1
        elif sent[1] == '1':
            actual_positive += 1;
        for word in sent[0]:
            for token in model:
                if token == word:
                    sent_prob.append(model[token])
        word_probs.append((sent[1],sent_prob))
    for sentence in word_probs:
        correct = sentence[0]
        summed,n = 0,0
        for val in sentence[1]:
            n += 1
            summed += val
        if n == 0:
            avg = 0.25
        else:
            avg = summed/n
        if avg >= 0.5 and correct == '1':
            correct_positive += 1
            result_positive += 1
        elif avg >= 0.5 and correct == '0':
            incorrect_positive += 1
            result_positive += 1
        elif avg < 0.5 and correct == '0':
            correct_negative += 1
            result_negative += 1
        elif avg < 0.5 and correct == '1':
            incorrect_negative += 1
            result_negative += 1
    test_print(actual_positive, actual_negative, correct_positive, correct_negative, 
               incorrect_positive, incorrect_negative, result_positive, result_negative)
            