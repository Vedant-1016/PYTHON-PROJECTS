import nltk
import pandas as pd
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import kagglehub
path = kagglehub.dataset_download("venky73/spam-mails-dataset")
df = pd.read_csv('spam_ham_dataset.csv')
print(df[['label', 'label_num']].drop_duplicates())
df['text'] = df['text'].apply(lambda x: x.replace('\r\n',' '))
stemmer = PorterStemmer()
corpus = []
stopwords_set = set(stopwords.words('english'))
for i in range(len(df)):
    text = df['text'].iloc[i].lower()
    text = text.translate(str.maketrans("","",string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    corpus.append(text)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
Y = df.label_num

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,Y_train)
print(clf.score(X_test,Y_test))

def spam_or_ham(email_content,classifier,vectorizer_model,stemmer_model,stopwords_set):
    processed_text = email_content.lower()
    processed_text = processed_text.replace('r/n/',' ')
    processed_text = processed_text.translate(str.maketrans("","",string.punctuation)).split()   
    """
    str.maketrans(x, y, z): This static method creates a translation table.
x: String of characters to be replaced.
y: String of characters to replace them with (must be same length as x).
z: String of characters to be deleted

e entire line processed_text = processed_text.translate(str.maketrans("","",string.punctuation)).split() effectively does this:

Creates a rule to delete all punctuation characters.
Applies that rule to the processed_text string, leaving only words and spaces.
Splits the clean string into a list of individual words, ready for further NLP steps like stemming and stop word removal.
"""
    processed_text = [stemmer_model.stem(word) for word in processed_text if word not in stopwords_set]
    processed_text = ''.join(processed_text)

    email_vector = vectorizer_model.transform([processed_text]).toarray()
    # Use .transform() not .fit_transform() as the vocabulary is already learned

    prediction = classifier.predict(email_vector)[0]

    if prediction==1:
        return 'Spam'
    else:
        return 'Ham'

#Example:
new_spam_email_1 = "Congratulations! You've won a free prize. Click this link now: www.fakegiveaway.com"
prediction_1 = spam_or_ham(new_spam_email_1, clf, vectorizer, stemmer, stopwords_set)
print(f"Email 1: '{new_spam_email_1}'\nPredicted: {prediction_1}\n")



