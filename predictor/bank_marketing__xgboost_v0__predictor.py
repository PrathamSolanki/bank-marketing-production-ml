import numpy as np
import pandas as pd
import xgboost
 
from google.cloud.aiplatform.utils import prediction_utils
from google.cloud.aiplatform.prediction.predictor import Predictor
 
class BankMarketingXGBoostPredictor(Predictor):
    def __init__(self):
        return

    def load(self, artifacts_uri: str):
        prediction_utils.download_model_artifacts(artifacts_uri)
        self._model = xgboost.Booster()
        self._model.load_model('model.json')
 
    def preprocess(self, prediction_input: dict):
        print(prediction_input)
        
        features_order_when_model_trained = self._model.feature_names
        
        if 'instances' in prediction_input and len(prediction_input) == 1:
            instances_df = pd.DataFrame(prediction_input['instances'][0], index=[0])
        else:
            instances_df = pd.DataFrame(prediction_input, index=[0])
        
        instances_df = instances_df[features_order_when_model_trained]
        print(instances_df.head())
        
        instances_df['default'] = instances_df['default'].apply(lambda x: True if x == 'yes' else False)
        instances_df['housing'] = instances_df['housing'].apply(lambda x: True if x == 'yes' else False)
        instances_df['loan'] = instances_df['loan'].apply(lambda x: True if x == 'yes' else False)
        
        instances_df['job'] = instances_df['job'].astype('category')
        instances_df['marital'] = instances_df['marital'].astype('category')
        instances_df['education'] = instances_df['education'].astype('category')
        instances_df['default'] = instances_df['default'].astype('boolean')
        instances_df['housing'] = instances_df['housing'].astype('boolean')
        instances_df['loan'] = instances_df['loan'].astype('boolean')
        instances_df['contact'] = instances_df['contact'].astype('category')
        instances_df['poutcome'] = instances_df['poutcome'].astype('category')
        instances_df['day_of_week'] = instances_df['day_of_week'].astype('category')
        instances_df['month'] = instances_df['month'].astype('category')
        
        instances_dmatrix = xgboost.DMatrix(instances_df, enable_categorical=True)
        return instances_dmatrix

    def predict(self, instances: xgboost.DMatrix):
        return self._model.predict(instances)

    def postprocess(self, prediction_results: np.ndarray):
        return {'predictions': prediction_results.tolist()}