from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import scipy as sp
import nltk
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from textblob import TextBlob, Word

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

import re
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

class EnsembleRegressor(BaseEstimator, ClassifierMixin):
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        self.pred_ = np.asarray([model.predict(X) for model in self.models])
        if self.weights:
            avg = np.average(self.pred_, axis=0, weights=self.weights)
        else:
            avg = np.average(self.pred_, axis=0)
        return avg


# -*- coding: utf-8 -*-
class SentimentMetrics(TransformerMixin):
    def changes (self, sentiments):
        N_changes = 0
        for i in range(len(sentiments)-1):
            if sentiments[i]*sentiments[i+1] < 0:
                N_changes+=1
        return N_changes     
    
    def sentiment(self, script, pages = 100, ma = 5):
        #Creates a moving average of the sentiment polarity of the script pages (or %)
        sentiment = []
        for i in range(pages+1-ma):
            sentiment.append(TextBlob(script[i*len(script)/pages:(i+ma)*len(script)/pages]).sentiment.polarity)
        return sentiment    
        
    def transform(self, X, **transform_params):  
        S = X.apply(lambda x: SentimentMetrics.sentiment(self, x))
        sentiment_dic = {
       'Range': S.apply(lambda x: max(x) - min(x)),
       #'Sentiment_avg': S.apply(lambda x: x.mean()),
       #'Sentiment_std': S.apply(lambda x: x.std()),
       'Max_sentiment': S.apply(lambda x: max(x)),
       'Max_sentiment_loc': S.apply(lambda x: x.index(max(x))),
       'Min_sentiment': S.apply(lambda x: min(x)),
       'Min_sentiment_loc': S.apply(lambda x: x.index(min(x))),
       'Max_pos_change': S.apply(lambda x: pd.Series(x).diff(1).max()),
       'Max_pos_loc': S.apply(lambda x: list(pd.Series(x).diff(1)).index(pd.Series(x).diff(1).max())),
       'Max_neg_change': S.apply(lambda x: pd.Series(x).diff(1).min()),
       'Max_neg_loc': S.apply(lambda x: list(pd.Series(x).diff(1)).index(pd.Series(x).diff(1).min())),
       'Begin_sentiment': S.apply(lambda x: x[0]),
       'End_sentiment': S.apply(lambda x: x[-1]),
       'Total_change': S.apply(lambda x: x[-1] - x[0]),
       'plot_twist_index': S.apply(lambda x: (max(x[-10:]) - min(x[-10:]))/(max(x[:-10]) - min(x[:-10]) + 0.01)),
       'pol_changes': S.apply(lambda x: SentimentMetrics.changes(self, x))
        }
            
        return sp.sparse.csr_matrix(pd.DataFrame(sentiment_dic).astype(float))
    
        
    
    def fit(self, X, y=None, **fit_params):
        return self      

class NamedEntities(TransformerMixin):        
    def number_entities(self, text):
        upper = [i for i in TextBlob(text).words if (i.isupper() and i.lower() not in stopwords.words('english'))]
        tagged = pos_tag(upper)
        return len(set([word for word,pos in tagged if pos == 'NNP']))
    
    def transform (self, X):
        return pd.DataFrame(X.apply(lambda x: NamedEntities.number_entities(self,x)))
    
    def fit(self, X, y=None, **fit_params):
        return self      
        
class ToSparse(TransformerMixin):        
    def transform (self, X):
        return sp.sparse.csr_matrix(X).T
        
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)    
    
    def fit(self, X, y=None, **fit_params):
        return self     

class SelectColumn(TransformerMixin):
    def __init__(self, column):
        self.column = column
        
    def transform(self, X, **transform_params):
        return X[self.column]
    
    def fit(self, X, y=None, **fit_params):
        return self      

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
def word_tokenize(text, how='lemma'):
    words = TextBlob(text).words
    if how == 'lemma':
        return [word.lemmatize() for word in words]
    elif how == 'stem':
        return [stemmer.stem(word) for word in words]
    
def sentence_tokenize(text):
    words = TextBlob(text.replace(',', '.')).sentences
    return [word for word in words]
   
from sklearn import decomposition

from sklearn.base import BaseEstimator
class DenseTransformer(TransformerMixin, BaseEstimator):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
        
        
class PlotSentiment():
    def __init__(self,script,pages):
        self.pages = pages
        self.script = script
        self.sentences = TextBlob(self.script).sentences
        self.text, self.y = self.sentiment(self.sentences, self.pages)
        
    def concatenate_sentences(self, sentences):
        s = ''
        for i in sentences:
            s += str(i) + '\n'
        return s.replace('  ', '')


    def sentiment (self, sentences_list, pages):
        n_sentences = len(sentences_list)/(pages)
        sentiment = []
        text = []
        for i in range(0,pages):
            temp =pd.Series(sentences_list[i*n_sentences:i*n_sentences+n_sentences]).apply(lambda x: x.sentiment.polarity)
            
            string = self.concatenate_sentences(sentences_list[i*n_sentences:i*n_sentences+n_sentences])
        
            sentiment.append(TextBlob(string).sentiment.polarity)   
            if TextBlob(string).sentiment.polarity>=0:
                idx = temp.idxmax()
            else:
                idx = temp.idxmin()
            text.append(str(sentences_list[i*n_sentences:i*n_sentences+n_sentences][idx]))    
            
       # temp = pd.Series(sentences_list[(pages + 1)*n_sentences:(pages+1)*n_sentences+n_sentences]).apply(lambda x: x.sentiment.polarity)    
        #sentiment.append(TextBlob(self.concatenate_sentences(sentences_list[(pages+1)*n_sentences:]) ).sentiment.polarity)  
        #if TextBlob(self.concatenate_sentences(sentences_list[(pages+1)*n_sentences:]) ).sentiment.polarity >=0:
        #    idx = temp.idxmax()
        #else:
        #    idx = temp.idxmin()
        #text.append(str(sentences_list[(pages+1)*n_sentences:][idx]))
        return text, pd.Series(sentiment)    

    def get_plot (self):
        y= list(self.y)
        x= range(len(list(self.y)))
        text=self.text

        text = [' '.join(s.split()) for s in text]
        return x,y,text