from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get data from form
        drug_name = request.form['drug_name']
        medical_condition = request.form['medical_condition']
        generic_name = request.form['generic_name']
        drug_classes = request.form['drug_classes']
        pregnancy_category = request.form['pregnancy_category']
        alcohol = request.form['alcohol']

        # Prepare input data for prediction
        input_data = np.array([[drug_name, medical_condition, generic_name, drug_classes, pregnancy_category, alcohol]])
        input_data_combined = [' '.join(input_data[0])]

        # Vectorize the input data
        input_vector = vectorizer.transform(input_data_combined)

        # Make prediction
        prediction = model.predict(input_vector)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
