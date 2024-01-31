import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import string

def cleanResume(text):
    text = re.sub('http\S+\s*', ' ', text) #removing urls
    text = re.sub('RT|cc', ' ', text) #removing RT or cc
    text = re.sub('#\S+', ' ', text) #removing hashtag
    text = re.sub('@\S+', ' ', text) #removing mentions
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text) #remving special characters
    text = re.sub(r'[^\x00-\x7f]', r' ', text) #removing non-ASCII value
    text = re.sub('\s+', ' ', text) #removing extra whitespace
    return text

def read_dataset():
    data = pd.read_csv("dataset/UpdatedResumeDataSet.csv")
    data1 = data.copy()
    data1['cleaned_resume'] = ''
    data1['cleaned_resume'] = data1['Resume'].apply(lambda x : cleanResume(x))
    data1.drop(columns = ['Resume'], inplace = True)
    data1.to_csv("dataset/cleaned_resume.csv", index = False)
    return data1


def most_common_words(df):
    stopWords = set(stopwords.words('english') + ["''"])
    totalWords = []
    sentences = df['cleaned_resume'].values
    cleanedSentences = ""
    for records in sentences:
        text = cleanResume(records)
        cleanedSentences += text
        requiredWords = nltk.word_tokenize(text)
        for word in requiredWords:
            if word not in stopWords and word not in string.punctuation:
                totalWords.append(word)
    frequency = nltk.FreqDist(totalWords)
    print(frequency.most_common(50))

    wc = WordCloud().generate(cleanedSentences)
    plt.figure(figsize = (8, 8))
    plt.imshow(wc, interpolation = 'bilinear')
    plt.axis('off')
    plt.show()