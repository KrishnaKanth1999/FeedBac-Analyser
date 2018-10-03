import pickle


class Loading:
    feature, target = None, None
    model = None


    def training_file_loading(self):
        X_in = open('../X.pickle', 'rb')
        y_in = open('../y.pickle', 'rb')

        self.feature = pickle.load(X_in)
        self.target = pickle.load(y_in)
        return  self.feature,self.target

    def models_file_loading(self):
        with open('tfidfmodel.pickle', 'rb') as f:
            self.tfidf = pickle.load(f)

        with open('classifier.pickle', 'rb') as f:
            self.clf = pickle.load(f)
        return self.tfidf,self.clf











