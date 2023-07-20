import pandas as pd
import sklearn
import nltk
#nltk.download_shell()
import string
from nltk.corpus import stopwords

reviews = pd.read_csv('cleaned_data.csv', sep=',')
print(reviews.head())

reviews = reviews.dropna()
# Convert 'verified_reviews' column to strings
reviews['verified_reviews'] = reviews['verified_reviews'].astype(str)





#corpus of strings to vector format --> bag of words

#removing punctuation and stopwords from the review text to have a clean data
def text_process(rev):
    nopunc = [char for char in rev if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Tokenizing the reviews with the created function
reviews['processed_reviews'] = reviews['verified_reviews'].apply(lambda rev: text_process(rev))

print(reviews['processed_reviews'])



"""vectorization (bag of words)
count how many times a word occur
weight the counts
normalize the vectors
"""

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(reviews['verified_reviews'])

print(len(bow_transformer.vocabulary_))

#testing if this method works: Seeing how many times each word repeated itself
rev2 = reviews['verified_reviews'][3]
print(rev2)
bow2 = bow_transformer.transform([rev2])
print(bow2)
print(bow2.shape)



#applying bow to whole data
rev_bow = bow_transformer.transform(reviews['verified_reviews'])
print('Shape of Sparse Matrix:', rev_bow.shape)
print(rev_bow.nnz)

#using tfidf transformer interpreting the cretaed vector as weight values:
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(rev_bow)

rev_tfidf = tfidf_transformer.transform(rev_bow)

tfidf2 = tfidf_transformer.transform(bow2)

#now training the ready data

from sklearn.naive_bayes import ComplementNB

rating_model = ComplementNB().fit(rev_tfidf, reviews['rating'])


print('predicted:', rating_model.predict(tfidf2)[0])
print('expected:', reviews.rating[3])

all_predictions = rating_model.predict(rev_tfidf)
print(all_predictions)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(reviews['rating'], all_predictions))
print('\n')
print(confusion_matrix(reviews['rating'], all_predictions))


#Trying how does the model predict a new review
review = "I hate it so much"
bow_review = bow_transformer.transform(text_process(review))
tfidf_review = tfidf_transformer.transform(bow_review)
prediction = rating_model.predict(tfidf_review)[0]
print(prediction)
