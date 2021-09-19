# IMDB sentiment with Vowpal Wabbit

Attempt to come up with [Vowpal Wabbit](https://vowpalwabbit.org/index.html) for the classic task of IMDB reviews sentiment classification ([Kaggle](https://www.kaggle.com/c/word2vec-nlp-tutorial/leaderboard)). Training was performed on the [IMDb Largest Review Dataset](https://www.kaggle.com/ebiswas/imdb-review-dataset), which originally comes in form of 7GB JSON files.

Achieved result on the test set (public LB): **0.9925** ROC AUC

Developed a simple api for deployment vw sentiment models.

## Notebooks

- #### Raw data preprocessing – [Kaggle kernel](https://www.kaggle.com/andrii0yerko/preprocessing-for-vowpal-wabbit-sentiment-analysis) – [nbviewer](https://nbviewer.jupyter.org/github/andrii0yerko/imdb-sentiment-with-vowpal-wabbit/blob/master/notebooks/preprocessing-for-vowpal-wabbit-sentiment-analysis.ipynb)

  Preprocessing consisted of json parsing, creating labels for binary sentiment, standard stop-words & non-words removing, stemming and saving the result in VW format.

- #### Model training – [Kaggle kernel](https://www.kaggle.com/andrii0yerko/imdb-sentiment-with-vowpal-wabbit) – [nbviewer](https://nbviewer.jupyter.org/github/andrii0yerko/imdb-sentiment-with-vowpal-wabbit/blob/master/notebooks/imdb-sentiment-with-vowpal-wabbit.ipynb)

  Trained SVM for binary classification, following hyperparameter tuning performed: comparison of different hash bit sizes, ngrams order and their combinations, the most appropriate one was `--bit_precision=28` and `--ngram=2 --ngram=3`, attempt to introduce a l1/l2 regularization, which was redundant for such a sparse feature space and didn't give a better result.

## Api server
Api developed with Flask and Docker, and based on sklearn wrapper of vowpal-wabbit.

> Notice that API developed assuming probability prediction, and the sigmoid function will be applied to the linear output.
> 
> Beware of learning your models with losses different than logistic, for the correct work.
### Models preparing:
Before deployment, Vowpal Wabbit models should be converted to the one used by the application, which can be done with `create_vw_pipeline` function from the `application.create_pipeline`.
```python
def create_vw_pipeline(vw_model_path, output_path=None, tag='0.0', comment=None)
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
```
Notice, that created pipelines should be placed in the `./models` folder, as it is done by default, and have a `.jl` extension.

### Docker image:
Create all the needed pipelines, and build the image with the Dockerfile.

Image includes Vowpal Wabbit installation as well, which can be used for training new models within the running container.
### Endpoints:
All the pipelines will be loaded on application start and the following endpoints produced for each of them:

- `GET /api/v{tag}/`

  **Response**: `200 OK`, `{"info": additional information (comment), "version": tag}`

- `POST /api/v{tag}/predict`
  
  **Request body**: ```{"text": "some text"}```

  **Response**: `200 OK`, `{"positive": probability, "negative": probability}`

- `POST /api/v{tag}/weight`
  
  **Request body**: ```{"text": "some text"}```

  **Response**: `200 OK`, `{"word1": weight, "word2": weight, "n gram1": weight, ...}`
