import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_preprocessing import *
from keywords import *

data = read_dataset()
var_mod = ['Category']
label_encoder = LabelEncoder()
for columns in var_mod:
    data[columns] = label_encoder.fit_transform(data[columns])

requiredText = data['cleaned_resume'].values
requiredTarget = data['Category'].values
wordVectorizer = TfidfVectorizer(sublinear_tf = True, stop_words = 'english')
wordVectorizer.fit(requiredText)
wordFeatures = wordVectorizer.transform(requiredText)
x_train, x_test, y_train, y_test = train_test_split(wordFeatures, requiredTarget, test_size = 0.3, random_state = 2000, shuffle = True, stratify = requiredTarget)
x_test, x_cv, y_test, y_cv = train_test_split(x_test, y_test, test_size=0.5, random_state=2000, shuffle=True, stratify=y_test)

def create_models():
    lr = LogisticRegression()
    nb = MultinomialNB(alpha = 0.01)
    svc = SVC(C = 1.0, kernel = 'linear', gamma = 'auto')
    return {"Logistic Regression": lr, "Naive Bayes": nb, "SVM": svc}

models = create_models()

metrcs = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
columns = ['Logistic Regression', 'Naive Bayes', 'SVM']
evaluation_df = pd.DataFrame(columns = columns, index = metrcs)

def training_models():
    for model_item in models.items():
        model = model_item[1]
        print(f"{model} is training")
        model.fit(x_train, y_train)
        
        # Training set performance
        train_prediction = model.predict(x_train)
        print(f"Training Set - {model_item[0]}:")
        print_metrics(y_train, train_prediction)
        
        # Test set performance
        test_prediction = model.predict(x_test)
        print(f"Test Set - {model_item[0]}:")
        print_metrics(y_test, test_prediction)
        
        # Cross-validation set performance
        cv_prediction = model.predict(x_cv)
        print(f"Cross-Validation Set - {model_item[0]}:")
        print_metrics(y_cv, cv_prediction)

        if str(model) == "MultinomialNB(alpha=0.01)":
            print(f"Naive Bayes Accuracy: {metrics.accuracy_score(y_test, test_prediction):.2f}\n")
        
        # Save the trained model
        filename = f'trained_models/{model_item[0]}.pkl'
        pickle.dump(model, open(filename, 'wb'))

def print_metrics(true_labels, predicted_labels):
    accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    precision = metrics.precision_score(true_labels, predicted_labels, average='macro')
    recall = metrics.recall_score(true_labels, predicted_labels, average='macro')
    f1 = metrics.f1_score(true_labels, predicted_labels, average='macro')
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}\n")

training_models()

filename = 'resume_vectorizer.pkl'
pickle.dump(wordVectorizer, open(f'vectorizer/{filename}', 'wb'))

def calculate_resume_scores(resume, job_category):
    resume = resume.lower()
    keywords = get_key_words()[job_category]
    resume_score = 0
    weight = 0
    for keyword, score in keywords.items():
        weight += score
        if keyword in resume:
            resume_score += score
    if int((resume_score / weight) * 100) <= 10:
        return 10
    else:
        return int((resume_score / weight) * 100)
