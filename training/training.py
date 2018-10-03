# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import pickle
from nltk.corpus import stopwords
from preprocessing.Preprocessing import Preprocessing
from LoadingFiles.Loading import Loading
stopwords=stopwords.words('english')
stopwords.remove('not')
print(stopwords)
#Data_Set Loading
obj=Loading()
X,y=obj.training_file_loading()

#PreProcessing
obj=Preprocessing()
features=[]
for i in X:
    text=obj.pre_process(i)
    features.append(text)

#Modeling

# Creating the Tf-Idf model directly


vectorizer = TfidfVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords)
X = vectorizer.fit_transform(features).toarray()

# Splitting the dataset into the Training set and Test set


text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training the classifier


classifier = MultinomialNB()
classifier.fit(text_train, sent_train)

# Testing model performance
sent_pred = classifier.predict_proba(text_test)

from sklearn.metrics import confusion_matrix

#cm = confusion_matrix(sent_test, sent_pred)

# Saving our classifier
with open('classifier.pickle', 'wb') as f:
    pickle.dump(classifier, f)

# Saving the Tf-Idf model
with open('tfidfmodel.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)

