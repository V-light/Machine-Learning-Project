#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from HR_FOR_API import pred
from flask import Flask , jsonify , request
from flask_restful import Resource , Api

app = Flask(__name__)

api = Api(app)

class HRPred(Resource):
    def post(self):
        data = request.get_json()
        res = pred(data)
        return {"Result": str(res)}
    
api.add_resource(HRPred, "/HR_pred")

if __name__ = = "__main__":
    app.run(debug = True)

