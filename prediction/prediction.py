from LoadingFiles.Loading import Loading
import re
from collections import Counter
import nltk
from operator import itemgetter
class prediction:
    def predict(self,text):
        #predicting
        obj=Loading()
        tfidf,clf=obj.models_file_loading()
        sample = tfidf.transform(text).toarray()
        sentiment = clf.predict_proba(sample)
        return sentiment

    def word_cloud(self):
        f = open("./feedbacks.txt", "r")
        contents = f.read()
        text = re.sub(r'\W', ' ', str(contents))
        text = text.lower()
        text = re.sub(r'^br$', ' ', text)
        text = re.sub(r'\s+br\s+', ' ', text)
        text = re.sub(r'\s+[a-z]\s+', ' ', text)
        text = re.sub(r'^b\s+', '', text)
        text = re.sub(r'\s+', ' ', text)
        words = nltk.word_tokenize(text)
        word_freqs = Counter(words)
        word_freqs = dict(word_freqs)

        word_freqs_js = []

        for key, value in word_freqs.items():
            temp = {"text": key, "size": value}
            word_freqs_js.append(temp)

        max_freq = max(word_freqs.values())
        word_freqs_js = sorted(word_freqs_js, key=itemgetter('size'), reverse=True)
        return word_freqs_js,max_freq