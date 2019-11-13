from titanicForAPI import pred
from flask import Flask, jsonify, request
from flask_restful import Resource, Api

# creating the flask app
app = Flask(__name__)
# creating an API object
api = Api(app)

class TitanicPred(Resource):
    def get(self):
        return jsonify({'message': 'hello world'})
    def post(self):
        data = request.get_json()
        res = pred(data)
        return {"result": str(res)}
api.add_resource(TitanicPred, "/titanic_pred")


if __name__ == '__main__':
    app.run(debug = True)
