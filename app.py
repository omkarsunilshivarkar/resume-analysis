from flask import render_template, Flask, request, render_template, redirect, url_for, send_from_directory

# for storing candidates info
from flask_sqlalchemy import SQLAlchemy
# new data
# from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import os
import pickle
from PyPDF2 import PdfReader
from pyresparser import ResumeParser
from train_model import *
from data_preprocessing import *
import warnings
import keywords
warnings.filterwarnings('ignore')

file_path = os.path.abspath(os.getcwd())+"\database.db"

# veryyy new 
# UPLOAD_FOLDER = '/path/to/the/uploads'
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
# .

app = Flask(__name__)


# important line
app.app_context().push()
# candidate info
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'+file_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# veryy new
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# .



db = SQLAlchemy(app)

# Define the SQLAlchemy model


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    company = db.Column(db.String(100), nullable=False)
    position = db.Column(db.String(100), nullable=False)
    resume_score = db.Column(db.Integer, nullable = False)
    files = db.Column(db.String(100),nullable= False)


# Create the database table
db.create_all()


@app.route('/')
def mainpage():
    return render_template('mainpage.html')


@app.route('/candidate', methods=['POST', 'GET'])
def candidate():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        company = request.form['company']
        position = request.form['position']

        # db.session.add(new_user)
        # db.session.commit()

        # code for storing resume into a folder
        uploaded_file = request.files.get('files', False)
        if uploaded_file:
            file_name = secure_filename(uploaded_file.filename)

            # new 
            # uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
            # new_user.cv = file_name 

            # veryyy new 
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            uploaded_file.save(file_path)
            read = PdfReader(file_path)
            resume = read.pages[0].extract_text()
            lr_model = pickle.load(open('trained_models/Logistic Regression.pkl', 'rb'))
            vectors = pickle.load(open('vectorizer/resume_vectorizer.pkl', 'rb'))
            cleaned_resume = cleanResume(resume)
            print(f'\nCleaned Resume:\n{cleaned_resume}')
            resume_data = ResumeParser(file_path).get_extracted_data()
            for skill in resume_data["skills"]:
                print(f'{skill},', end = ' ')
            print()
            input = vectors.transform([resume])
            lr_prediction_id = lr_model.predict(input)[0]
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
            print(f'\nPrediction from Logistic Regression: {category_map.get(lr_prediction_id)}')

            score = calculate_resume_scores(cleaned_resume, category_map.get(lr_prediction_id))
            print(f'\nResume score: {score}/100')

            new_user = User(fullname=fullname, email=email,
                            company=company, position=position,files=file_path, resume_score = score)
            # .


            # uploaded_file.save(os.path.join(
            #     'C:/Users/ADMIN/Desktop/RESUME ANALYSIS WEBSITE 01.11.23/my_flask_app', file_name))
            # print('Resume uploaded successfully')

            # new 
            db.session.add(new_user)
            db.session.commit()
        else:
            print('No file selected')
           # return render_template('candidate.html')
        return render_template('result.html', pred = category_map.get(lr_prediction_id), score = score)
    return render_template('candidate.html')


@app.route('/company')
def company():
    users = User.query.all()  # Retrieve all user records from the database
    # return render_template('users.html', users=users)
    return render_template('company.html',users=users)


# new 

@app.route('/files/<file_name>')
def serve_cv(file_name):
    actual_file = file_name.split('\\')[1]
    return send_from_directory(app.config['UPLOAD_FOLDER'], actual_file)



@app.route('/howitworks')
def howitworks():
    return render_template('howitworks.html')


@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
