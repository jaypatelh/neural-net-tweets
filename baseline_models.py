import numpy as np 
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

def train_model(training_data, labels, model_type):
	if model_type == 'nb':
		clf = MultinomialNB()
	elif model_type == 'svm':
		clf = svm.SVC()
	else:
		clf = RandomForestClassifier()
	clf.fit(training_data, labels)
	return clf

def calculate_score(test_data, test_labels, model):
	return model.score(test_data, test_labels)

def predict_labels(test_data, model):
	return model.predict(test_data)

