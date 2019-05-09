from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


params = {
    'vect__binary': True,
    'vect__ngram_range': (1, 5),
    'vect__min_df': 3,
    'vect__max_df': 0.9,
    'clf__random_state': 0,
}

def build_pipeline():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', LogisticRegression(random_state=0)),
    ])
    
    pipeline.set_params(**params)
    
    return pipeline
