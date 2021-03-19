from sklearn.feature_extraction.text import TransformerMixin

class DenseTransformer(TransformerMixin):
    '''
    Custom class inherited from sklearn TransformerMixin.
    Used for transform the sparse matrix used in sklearn
    TF-IDF vectorizer in the pipeline model.
    '''
    
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()
