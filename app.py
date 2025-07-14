from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load dataset
df = pd.read_csv("cirrhosis.csv")

# Drop ID, Stage, and Status from form inputs
for col in ['ID', 'Stage', 'Status']:
    if col in df.columns:
        df = df.drop(columns=[col])

input_columns = df.columns.tolist()

# Load both models
with open("model_stage.pkl", "rb") as f:
    model_stage = pickle.load(f)

with open("model_status.pkl", "rb") as f:
    model_status = pickle.load(f)

@app.route('/')
def index():
    return render_template("index.html", columns=input_columns)

@app.route('/predict', methods=['POST'])
def predict():
    form_data = [request.form[col] for col in input_columns]
    input_df = pd.DataFrame([form_data], columns=input_columns)

    # Convert numeric values
    for col in input_df.columns:
        try:
            input_df[col] = pd.to_numeric(input_df[col])
        except:
            pass

    # Predict both targets
    pred_stage = model_stage.predict(input_df)[0]
    pred_status = model_status.predict(input_df)[0]

    return render_template("index.html", columns=input_columns,
                           prediction_stage=pred_stage,
                           prediction_status=pred_status,
                           values=request.form)

@app.route('/suggest', methods=['GET'])
def suggest():
    row = df.sample(1).iloc[0]
    suggested = {col: str(row[col]) for col in input_columns}
    return jsonify(suggested)

if __name__ == '__main__':
    app.run(debug=True)
