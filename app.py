from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
application = Flask(__name__)

app=application

#home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict_marks',methods=['GET','POST'])
def predict_marks():
    if request.method == 'GET':
        return render_template('home.html')
    elif request.method=='POST':
        
        gender = request.form.get('gender')
        race_ethnicity = request.form.get('race_ethnicity')
        parental_level_of_education = request.form.get('parental_level_of_education')
        lunch = request.form.get('lunch')
        test_preparation_course = request.form.get('test_preparation_course')
        reading_score = request.form.get('reading_score')
        writing_score = request.form.get('writing_score')

        new_data = CustomData(
            gender = gender, 
            race_ethnicity = race_ethnicity,
            parental_level_of_education = parental_level_of_education,
            lunch = lunch,
            test_preparation_course = test_preparation_course,
            reading_score = reading_score,
            writing_score = writing_score
        )

        new_data_df = new_data.get_data_as_frame()

        predictor = PredictPipeline()
        result = predictor.predict(new_data_df)

        return render_template('home.html',results=result[0])

if __name__=='__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

