import pickle
import xgboost
from flask import Flask
from flask import request
from flask import jsonify

with open("xg_boost_model.bin", "rb") as f_in:
    dv, model=pickle.load(f_in)


app = Flask('depression')

@app.route('/predict', methods=['POST'])
def predict():
    student = request.get_json()

    X = dv.transform([student])
    X=xgboost.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
    y_pred = model.predict(X)
    depression = y_pred >= 0.5

    result = {
        'depression_probability': float(y_pred[0]),
        'depression': bool(depression)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)


