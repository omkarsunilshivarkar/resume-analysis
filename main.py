import pickle
from train_model import *
from PyPDF2 import PdfReader
from pyresparser import ResumeParser
import warnings
import keywords
warnings.filterwarnings('ignore')

read = PdfReader('Sample Resume/Document.pdf')
resume = read.pages[0].extract_text()
lr_model = pickle.load(open('trained_models/Logistic Regression.pkl', 'rb'))
nb_model = pickle.load(open('trained_models/Naive Bayes.pkl', 'rb'))
svm_model = pickle.load(open('trained_models/SVM.pkl', 'rb'))
vectors = pickle.load(open('vectorizer/resume_vectorizer.pkl', 'rb'))
cleaned_resume = cleanResume(resume)
print(f'\nCleaned Resume:\n{cleaned_resume}')
resume_data = ResumeParser('Sample Resume/Document.pdf').get_extracted_data()
print('\nYour skills are:')
for skill in resume_data["skills"]:
    print(f'{skill},', end = ' ')
print()

input = vectors.transform([resume])
lr_prediction_id, nb_prediction_id, svm_prediction_id = lr_model.predict(input)[0], nb_model.predict(input)[0], svm_model.predict(input)[0]
category_map = {15 :'Java Developer',
    23: 'Testing',
    8: 'DevOps Engineer',
    20: 'Python Developer',
    24: 'Web Designing',
    12: 'HR',
    13: 'Hadoop',
    3: 'Blockchain',
    10: 'ETL Developer',
    18: 'Operations Manager',
    6: 'Data Science',
    22: 'Sales',
    16: 'Mechanical Engineer',
    1: 'Arts',
    7: 'Database',
    11: 'Electrical Engineering',
    14: 'Health and fitness',
    19: 'PMO',
    4: 'Business Analyst',
    9: 'DotNet Developer',
    2: 'Automation Testing',
    17: 'Network Security Engineer',
    21: 'SAP Developer',
    5: 'Civil Engineer',
    0: 'Advocate'
    }
# print(f'\nPrediction from Logistic Regression: {category_map.get(lr_prediction_id)}',
#       f'Prediction from Naive Bayes: {category_map.get(nb_prediction_id)}',
#       f'Prediction from SVM: {category_map.get(svm_prediction_id)}', sep = '\n')

print(f'\nPrediction from Logistic Regression: {category_map.get(lr_prediction_id)}')

#keyword = keywords.get_key_words()
score = calculate_resume_scores(cleaned_resume, category_map.get(lr_prediction_id))
print(f'\nResume score: {score}/100')

