import pickle

from sklearn.datasets import load_files


class Generate:
    def __init__(self):
        self.load_file()

    def load_file(self):
        # Importing the dataset
        reviews = load_files('../files')
        X, y = reviews.data, reviews.target
        # Pickling the dataset
        with open('X.pickle', 'wb') as f:
            pickle.dump(X, f)

        with open('y.pickle', 'wb') as f:
            pickle.dump(y, f)

        print("hi")



