import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier 
import seaborn as sns
import pickle
from sklearn.impute import SimpleImputer

data = pd.read_csv('C:/Users/brids/Documents/my_flask_app/my_flask_app_1/dataset/data-final.csv', sep='\t')
data = data.copy()
data.drop(data.columns[50:107], axis=1, inplace=True)
data.drop(data.columns[51:], axis=1, inplace=True)
print('Number of participants:', len(data))

# Define personality questions
ext_questions = {'EXT1' : 'I am the life of the party',
                 'EXT2' : 'I dont talk a lot',
                 'EXT3' : 'I feel comfortable around people',
                 'EXT4' : 'I keep in the background',
                 'EXT5' : 'I start conversations',
                 'EXT6' : 'I have little to say',
                 'EXT7' : 'I talk to a lot of different people at parties',
                 'EXT8' : 'I dont like to draw attention to myself',
                 'EXT9' : 'I dont mind being the center of attention'}

est_questions = {'EST1' : 'I get stressed out easily',
                 'EST2' : 'I am relaxed most of the time',
                 'EST3' : 'I worry about things',
                 'EST4' : 'I seldom feel blue',
                 'EST5' : 'I am easily disturbed',
                 'EST6' : 'I get upset easily',
                 'EST7' : 'I change my mood a lot',
                 'EST8' : 'I have frequent mood swings',
                 'EST9' : 'I get irritated easily',
                 'EST10': 'I often feel blue'}

agr_questions = {'AGR1' : 'I feel little concern for others',
                 'AGR2' : 'I am interested in people',
                 'AGR3' : 'I insult people',
                 'AGR4' : 'I sympathize with others feelings',
                 'AGR5' : 'I am not interested in other peoples problems',
                 'AGR6' : 'I have a soft heart',
                 'AGR7' : 'I am not really interested in others',
                 'AGR8' : 'I take time out for others',
                 'AGR9' : 'I feel others emotions',
                 'AGR10': 'I make people feel at ease'}

csn_questions = {'CSN1' : 'I am always prepared',
                 'CSN2' : 'I leave my belongings around',
                 'CSN3' : 'I pay attention to details',
                 'CSN4' : 'I make a mess of things',
                 'CSN5' : 'I get chores done right away',
                 'CSN6' : 'I often forget to put things back in their proper place',
                 'CSN7' : 'I like order',
                 'CSN8' : 'I shirk my duties',
                 'CSN9' : 'I follow a schedule'}

opn_questions = {'OPN1' : 'I have a rich vocabulary',
                 'OPN2' : 'I have difficulty understanding abstract ideas',
                 'OPN3' : 'I have a vivid imagination',
                 'OPN4' : 'I am not interested in abstract ideas',
                 'OPN5' : 'I have excellent ideas',
                 'OPN6' : 'I do not have a good imagination',
                 'OPN7' : 'I am quick to understand things',
                 'OPN8' : 'I use difficult words',
                 'OPN9' : 'I spend time reflecting on things',
                 'OPN10': 'I am full of ideas'}

EXT = [column for column in data if column.startswith('EXT')]
EST = [column for column in data if column.startswith('EST')]
AGR = [column for column in data if column.startswith('AGR')]
CSN = [column for column in data if column.startswith('CSN')]
OPN = [column for column in data if column.startswith('OPN')]

train_size = int(0.7 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data.drop('country', axis=1))
test_data_scaled = scaler.transform(test_data.drop('country', axis=1))

imputer = SimpleImputer(strategy='mean')
train_data_scaled_imputed = imputer.fit_transform(train_data_scaled)
test_data_scaled_imputed = imputer.transform(test_data_scaled)

kmeans = KMeans(n_clusters=5, random_state=0)
k_fit = kmeans.fit(train_data_scaled_imputed)

test_predictions = k_fit.predict(test_data_scaled_imputed)

pca = PCA(n_components=2)
pca_fit = pca.fit_transform(train_data_scaled_imputed)
xgb_classifier = XGBClassifier(random_state=0)
xgb_classifier.fit(pca_fit, k_fit.labels_)


test_data_pca = pca.transform(test_data_scaled_imputed)
test_xgb_predictions = xgb_classifier.predict(test_data_pca)

cluster_labels_map = {
    0: 'Openness',
    1: 'Neuroticism',
    2: 'Agreeableness',
    3: 'Conscientiousness',
    4: 'Extraversion',
}
test_cluster_array = k_fit.predict(test_data_scaled_imputed)
print(f'Accuracy: {accuracy_score(test_cluster_array, test_xgb_predictions)}')
print(f'\nConfusion matrix:\n{confusion_matrix(test_cluster_array, test_xgb_predictions)}')
print(f'\nClassification report:\n{classification_report(test_cluster_array, test_xgb_predictions)}')

pickle.dump(xgb_classifier, open('C:/Users/brids/Documents/my_flask_app/my_flask_app_1/trained_models/Personality.pkl', 'wb'))

df_pca_test = pd.DataFrame(data=test_data_pca, columns=['PCA1', 'PCA2'])
df_pca_test['Clusters'] = test_predictions

plt.figure(figsize=(10, 10))
sns.scatterplot(data=df_pca_test, x='PCA1', y='PCA2', hue='Clusters', palette='Set2', alpha=0.9)
plt.title('Personality Clusters after PCA (Test Set)')
plt.show()
