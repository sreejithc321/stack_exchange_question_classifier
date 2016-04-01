'''
Stack Exchange Question Classifier
Given a question and an excerpt identify which topic it belongs to.

'''

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import json

def read_data(train_data):
    '''
    Read data from file and return as list 
    '''
    data = []
    with open(train_data) as f:
        for line in f:
            try:
                data.append(json.loads(line.lower()))
            except:
                data.append(line.strip('\n'))
    return data

def get_features(data):
    '''
    - Transform non-numerical 'topics' to numerical 'labels'
    - Convert ''contents' to a matrix of TF-IDF features
    '''
    topics = [ d['topic'] for d in data]
    contents =   [ d['question'] + ' ' + d['excerpt']  for d in data]
    labels = le.fit_transform(topics)   
    features = vectorizer.fit_transform(contents)
    return labels, features

def build_model(labels, features, algorithm):
    '''
    Train the model
    '''
    model = algorithm.fit(features, labels)
    return model


le = LabelEncoder()
vectorizer = TfidfVectorizer(max_df=1.0, ngram_range=(1,1),stop_words='english', use_idf='True')
algorithm = MultinomialNB()

## Train Data
train_data = 'data/training.json'
data = read_data(train_data)
labels, features = get_features(data[1:])

## Model
model = build_model(labels,features,algorithm)

## Test Data
test_data = read_data('data/test_input.txt')
test_labels = read_data('data/test_output.txt')
test_contents =  [ d['question'] + ' ' + d['excerpt']  for d in test_data[1:]]
test_features = vectorizer.transform(test_contents)

## Predict
predicted_topics = model.predict(test_features)
pred_labels = []
for topics in predicted_topics:
    pred_labels.append(le.inverse_transform(topics))

print 'Accuracy : ', accuracy_score(test_labels,pred_labels)
