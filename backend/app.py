from copy import deepcopy
import numbers
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

SEEDED_USERS = {
    "1": {"id": "1", "first_name": "Ava", "user_group": 11},
    "2": {"id": "2", "first_name": "Ben", "user_group": 22},
    "3": {"id": "3", "first_name": "Chloe", "user_group": 33},
    "4": {"id": "4", "first_name": "Diego", "user_group": 44},
    "5": {"id": "5", "first_name": "Ella", "user_group": 55},
}

MODEL_PATH = Path(__file__).resolve().parent / "src" / "random_forest_model.pkl"
PREDICTION_COLUMNS = [
    "city",
    "province",
    "latitude",
    "longitude",
    "lease_term",
    "type",
    "beds",
    "baths",
    "sq_feet",
    "furnishing",
    "smoking",
    "cats",
    "dogs",
]

app = Flask(__name__)
# For this lab, allow cross-origin requests from the React dev server.
# This broad setup keeps local development simple and is not standard
# production practice.
CORS(app)
users = deepcopy(SEEDED_USERS)

# TODO: Define these Flask routes with @app.route():
@app.route('/users', methods=['GET'])
def get_users():
    return jsonify(list(users.values())), 200


@app.route('/users', methods=['POST'])
def postUsers():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"message": "Request body is missing or empty."}), 400

    if 'id' not in data or 'first_name' not in data or 'user_group' not in data:
        return jsonify({"message": "Missing id, first_name, or user_group."}), 400

    user_id = str(data['id']).strip()
    first_name = str(data['first_name']).strip()

    if user_id == "" or first_name == "":
        return jsonify({"message": "id and first_name cannot be empty."}), 400

    if user_id in users:
        return jsonify({"message": f"User {user_id} already exists."}), 409

    users[user_id] = {
        "id": user_id,
        "first_name": first_name,
        "user_group": data['user_group']
    }

    return jsonify({
        "id": user_id,
        "first_name": first_name,
        "user_group": data['user_group'],
        "message": f"Created user {user_id}."
    }), 201

@app.route('/users/<user_id>', methods=['PUT'])
def update_user(user_id):
    data = request.get_json()

    if user_id not in users:
        return jsonify({"message": f"User {user_id} was not found."}), 404

    if not data or 'first_name' not in data or 'user_group' not in data:
        return jsonify({"message": "Missing first_name or user_group."}), 400

    users[user_id] = {
        "id": user_id,
        "first_name": data['first_name'],
        "user_group": data['user_group']
    }

    return jsonify({
        "id": user_id,
        "first_name": data['first_name'],
        "user_group": data['user_group'],
        "message": f"Updated user {user_id}."
    }), 200


@app.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    if user_id not in users:
        return jsonify({"message": f"User {user_id} was not found."}), 404

    del users[user_id]
    return jsonify({"message": f"Deleted user {user_id}."}), 200



#   Exercise2
# - POST /predict_house_price
@app.route('/predict_house_price', methods=['POST'])
def predict_house_price():
     model = joblib.load(MODEL_PATH)
     data = request.json
     cats = data['pets']
     dogs = data['pets']
     sample_data = [data['city'], data['province'], float(data['latitude']), float(data['longitude']), data['lease_term'], data['type'], float(data['beds']), float(data['baths']), float(data['sq_feet']), data['furnishing'], data['smoking'], cats, dogs]
     sample_df = pd.DataFrame([sample_data], columns=['city', 'province', 'latitude', 'longitude', 'lease_term', 'type', 'beds', 'baths', 'sq_feet', 'furnishing', 'smoking', 'cats', 'dogs'])
     predicted_price = model.predict(sample_df)
     if not isinstance(data['city'], str) or not data['city'].isalpha():
          return jsonify({"message": "City needs to be words"}), 400
     elif isinstance(data['longitude'], numbers.Number):
          return jsonify({"message": "longitude needs to be a number"}), 400
     elif isinstance(data['longitude'], numbers.Number):
          return jsonify({"message": "longitude needs to be a number"}), 400
     elif isinstance(data['beds'], numbers.Number):
          return jsonify({"message": "beds needs to be a number"}), 400
     elif isinstance(data['baths'], numbers.Number):
          return jsonify({"message": "baths needs to be a number"}), 400
     elif isinstance(data['sq_feet'], numbers.Number):
          return jsonify({"message": "sq_feet needs to be a number"}), 400
     return jsonify({"predicted_price": f"{predicted_price[0]}"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5050)
