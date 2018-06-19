from flask import Flask
from flask_restful import Resource, Api, reqparse
from webargs import fields, validate
from webargs.flaskparser import use_kwargs, parser
from flask import request
from sklearn.linear_model import SGDClassifier
import pickle


app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
      parser = reqparse.RequestParser()
      args = request.args
      loaded_model = pickle.load(open('/home/cxmdev/Predictive/Pred/finalized_model.sav', 'rb'))
      gender = args['gender']
      simi = args['similarity']
      # print str(gender) 
      # print "Male"
      # print str(gender) == "Male"
      if str(gender) == "Male":
      	val = 1
      elif gender == "Female":
      	val = 0
      # print type(simi)
      prob = loaded_model.predict_proba([[val, float(str(simi))]])[0][1]
      return {'ctrprob': prob}

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=3134)