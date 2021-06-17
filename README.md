# IMDB sentiment with Vowpal Wabbit

Attempt to come up with [Vowpal Wabbit](https://vowpalwabbit.org/index.html) for the classic task of IMDB reviews sentiment classification ([Kaggle](https://www.kaggle.com/c/word2vec-nlp-tutorial/leaderboard)). Training was performed on the [IMDb Largest Review Dataset](https://www.kaggle.com/ebiswas/imdb-review-dataset), which originally comes in form of 7GB JSON files.

Achieved result on the test set: 0.9724 accuracy score

### Notebooks

- Raw data preprocessing • [Kaggle kernel](https://www.kaggle.com/andrii0yerko/preprocessing-for-vowpal-wabbit-sentiment-analysis) • [nbviewer](https://nbviewer.jupyter.org/github/andrii0yerko/imdb-sentiment-with-vowpal-wabbit/blob/master/preprocessing-for-vowpal-wabbit-sentiment-analysis.ipynb)

  Preprocessing consisted of json parsing, creating labels for binary sentiment, standard stop-words & non-words removing, stemming and saving the result in VW format.

- Model training • [Kaggle kernel](https://www.kaggle.com/andrii0yerko/imdb-sentiment-with-vowpal-wabbit) • [nbviewer](https://nbviewer.jupyter.org/github/andrii0yerko/imdb-sentiment-with-vowpal-wabbit/blob/master/imdb-sentiment-with-vowpal-wabbit.ipynb)

  Trained SVM for binary classification, following hyperparameter tuning performed: comparison of different hash bit sizes, ngrams order and their combinations, the most appropriate one was `--bit_precision=28` and `--ngram=2 --ngram=3`, attempt to introduce a l1/l2 regularization, which was redundant for such a sparse feature space and didn't give a better result.

