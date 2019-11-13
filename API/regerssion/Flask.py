#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from HouseForAPI import pred
from flask import flask , jsonify, request
from flask_restful import Resource ,Api

app = Flask(__name__)
api = App(app)
class HousePred(Resource):
        def post(self):
        data = request.get_json()
        res = pred(data)
        return {"Result": str(res)}
    
api.add_resource(HousePred, "/HousePred")

if __name__ = = "__main__":
    app.run(debug = True)
    

