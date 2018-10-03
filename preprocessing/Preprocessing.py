import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords

class Preprocessing:

    def pre_process(self, text):
        text = re.sub(r'\W', ' ', str(text))
        text = text.lower()
        text = re.sub(r'^br$', ' ', text)
        text = re.sub(r'\s+br\s+', ' ', text)
        text = re.sub(r'\s+[a-z]\s+', ' ', text)
        text = re.sub(r'^b\s+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text


