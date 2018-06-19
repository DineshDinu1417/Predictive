from gensim.models import Word2Vec
import gensim, logging
from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint   # pretty-printer
import os, glob
from collections import defaultdict
from pprint import pprint
import re
from gensim.models import word2vec
from gensim.models import Word2Vec
import numpy as np
# from sklearn.cluster import KMeans
import time
from multiprocessing import Pool
from numpy import dot
from numpy.linalg import norm
from flask import Flask
from flask_restful import Resource, Api, reqparse
from webargs import fields, validate
from webargs.flaskparser import use_kwargs, parser
from flask import request

# Set values for various parameters
num_features = 400   # Word vector dimensionality                      
min_word_count = 20   # Minimum word count                        
num_workers = 6       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = .00001   # Downsample setting for frequent words


model = Word2Vec.load("300features_40minwords_10context")
# print model.most_similar("female")
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word)

    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])

    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
       if counter%1000. == 0.:
        print "Review %d of %d" % (counter, len(reviews))
       reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features)

       counter = counter + 1.
    return 
    
def text_to_wordlist( review, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
      parser = reqparse.RequestParser()
      args = request.args
      campstr = args['campaignstring']
      intrtag = args['interestedtags']
      a = makeFeatureVec(text_to_wordlist(campstr),model,num_features)
      b = makeFeatureVec(text_to_wordlist(intrtag),model,num_features)
      cos_sim = dot(a, b)/(norm(a)*norm(b))
      return {'similarity': str(cos_sim)}

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True)

