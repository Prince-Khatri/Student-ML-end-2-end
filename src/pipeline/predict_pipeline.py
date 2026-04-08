from src.decorators import handle_exception
from src.logger import logging
import pandas as pd
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    @handle_exception
    def predict(self, features):
        logging.info("Starting Prediction Pipeline...")

        logging.info("Loading models...")
        model_path = 'artifacts/model.pkl'
        preprocessor_path = 'artifacts/preprocessor.pkl'
        
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)
        logging.info("Models Loaded...")

        logging.info("Transforming and predicting...")
        data_scaled = preprocessor.transform(features)
        preds = model.predict(data_scaled)

        logging.info("Prediction Pipeline Executiong Completed...")
        return preds


class CustomData:
    def __init__(self,
        gender, 
        race_ethnicity,
        parental_level_of_education,
        lunch,
        test_preparation_course,
        reading_score,
        writing_score):

        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    @handle_exception
    def get_data_as_frame(self):
        data_dict = {
            'gender':[self.gender],
            'race/ethnicity':[self.race_ethnicity],
            'parental level of education':[self.parental_level_of_education],
            'lunch':[self.lunch],
            'test preparation course':[self.test_preparation_course],
            'reading score':[self.reading_score],
            'writing score':[self.writing_score]
        }
        df = pd.DataFrame(data_dict)
        return df
        
    

