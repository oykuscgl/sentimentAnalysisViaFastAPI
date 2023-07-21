import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Understand and inspect the data
rev_data = pd.read_csv('cleaned_data.csv')

print(rev_data.head())
rev_data = rev_data.dropna()
print(rev_data.isnull().value_counts())

sns.countplot(data=rev_data, x='rating')
plt.show()

print(rev_data['rating'].value_counts())

rev_data['text_length'] = rev_data['verified_reviews'].apply(len)

print(rev_data.head())
print(rev_data.columns)

plt.tight_layout()
plt.figure(figsize=(12,6))
g = sns.FacetGrid(data=rev_data, col='rating')
g.map(plt.hist, 'text_length')
plt.show()

sns.barplot(data=rev_data, x='rating', y='text_length')
plt.show()

#Create your model
data_class = rev_data[(rev_data['rating'] == 1) | (rev_data['rating']==5)]
print(data_class.info())

X = data_class['verified_reviews']
y = data_class['rating']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB

pipeline = Pipeline([
    ('bow', CountVectorizer(lowercase=False)),
    ('classifier', ComplementNB())
])


X = data_class['verified_reviews']
y = data_class['rating']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 101)

pipeline.fit(X_train, y_train)

predict_MNV = pipeline.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(predict_MNV, y_test))
print(classification_report(predict_MNV, y_test))


review = "I love love lvoe lovee."
prediction = pipeline.predict([review])
print(prediction)





