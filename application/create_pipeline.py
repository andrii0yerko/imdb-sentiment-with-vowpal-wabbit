from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from vowpalwabbit.sklearn_vw import VWClassifier
import joblib


class ReviewPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.stops = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        super().__init__()

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        # if isinstance(X, str):
        #     return self.preprocess_review_(X)
        # else:
        X = np.atleast_1d(X).flatten()
        return np.array([self.preprocess_review_(x) for x in X])

    def preprocess_review_(self, raw_review):
        # Remove HTML
        review_text = BeautifulSoup(raw_review, features="html.parser").get_text()
        # Remove URLs
        review_text = re.sub(r"https?:\/\/[\w+.\/]+", " ", review_text)
        # Remove non-letters
        letters_only = re.sub(r"[^a-zA-Z]", " ", review_text)
        # Convert to lower case, split into individual words
        words = letters_only.lower().split()
        # Remove stop words (and stem others if needed)
        meaningful_words = [self.stemmer.stem(w) for w in words if w not in self.stops]
        return(" ".join(meaningful_words))

    def mark_up_text(self, review, max_ngram_order=1):
        idx = []
        ngrams, ngrams_idx = [], []
        words = self.preprocess_review_(review).split()
        review_lower = review.lower()

        # mark up a original text for stemmed words
        last_idx = 0
        for word in words:
            span = re.search(word+r"[a-z]*", review_lower[last_idx:]).span()
            span = [i+last_idx for i in span]
            idx.append(span)
            last_idx = span[1]
        # create ngrams
        for order in range(2, max_ngram_order+1):
            num = len(words) - max_ngram_order + 1
            for i in range(num):
                ngrams.append(" ".join(words[i: i+order]))
                ngrams_idx.append([idx[i][0], idx[i+order-1][1]])

        slices = [slice(*i) for i in idx+ngrams_idx]
        return [review[s] for s in slices], words + ngrams


class VWFeatureSpaceAdder(BaseEstimator, TransformerMixin):

    def __init__(self, feature_space='text'):
        self.feature_space = feature_space
        self.add_feature_space = np.vectorize(self.add_feature_space_)
        super().__init__()

    def add_feature_space_(self, x):
        return f'|{self.feature_space} ' + x

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        return self.add_feature_space(X)


def create_vw_pipeline(vw_model_path, output_path=None, tag='0.0', comment=None):
    '''
    Creates and serializes application pipeline from the Vowpal Wabbit model file.

    Parameters
    ----------
    vw_model_path : str
        Path to file of vowpal wabbit saved model
    output_path : str, optional
        Path where pipeline will be saved
        Extension of the output file will be .jl
        Default is 'models/pipeline-v{tag}'
    tag : str, optional
        The version of the outputting pipeline
        Default is '0.0'
    comment : str, optional
        Any additional information, that will be added to resulting file
        Default is None
        
    Produced file is a joblib dump of a dictionary in the following format
    {
        'pipeline': pipeline  # sklearn.pipeline.Pipeline
                              # containing preprocessing transformers
                              # and the classifier created from the vw file
        'tag': tag,
        'comment': comment
    }
    '''
    if output_path is None:
        output_path = f"models/pipeline-v{tag}"

    pipeline = Pipeline([
        ('preprocessor', ReviewPreprocessor()),
        ('namespace_adder', VWFeatureSpaceAdder()),
        ('classifier', VWClassifier(
            initial_regressor=vw_model_path, convert_to_vw=False
            ).fit()
         )
    ])

    model_dict = {
        'pipeline': pipeline,
        'tag': tag,
        'comment': comment
    }
    joblib.dump(model_dict, output_path+'.jl')
