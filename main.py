from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('cleaned_data.csv')
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("RidgeModel.pkl", "rb") as f:
    model = pickle.load(f)
@app.route('/')
def index():
    locations = sorted(data["location"].unique())
    return render_template('index.html', locations=locations)
@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))
    sqft = float(request.form.get('sqft'))
    input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                            columns=['location', 'total_sqft', 'bath', 'bhk'])

    try:
        input_transformed = preprocessor.transform(input_df)
        prediction = model.predict(input_transformed)[0]
        prediction = round(prediction, 2)
        prediction_text = f"🏠 Estimated House Price: ₹{prediction:,.2f}"
    except Exception as e:
        prediction_text = f"Error in prediction: {e}"
    locations = sorted(data["location"].unique())
    return render_template('index.html', locations=locations, prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
