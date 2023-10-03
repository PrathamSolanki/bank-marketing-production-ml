import os
from google.cloud import storage
from google.cloud import bigquery
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost

PROJECT_ID = <your project id>
bq_client = bigquery.Client(project=PROJECT_ID)

query = """
select *
from <bigquery dataset name>.<bigquery table name>;
"""
df = bq_client.query(query).to_dataframe()


features_list = df.columns.to_list()
features_list.remove('y')
x = df[features_list]
y = df[['y']]


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)


y_train['target'] = y_train['y'].apply(lambda x: True if x == 'yes' else False)
y_val['target'] = y_val['y'].apply(lambda x: True if x == 'yes' else False)


x_train['default'] = x_train['default'].apply(lambda x: True if x == 'yes' else False)
x_train['housing'] = x_train['housing'].apply(lambda x: True if x == 'yes' else False)
x_train['loan'] = x_train['loan'].apply(lambda x: True if x == 'yes' else False)

x_val['default'] = x_val['default'].apply(lambda x: True if x == 'yes' else False)
x_val['housing'] = x_val['housing'].apply(lambda x: True if x == 'yes' else False)
x_val['loan'] = x_val['loan'].apply(lambda x: True if x == 'yes' else False)


x_train['job'] = x_train['job'].astype('category')
x_train['marital'] = x_train['marital'].astype('category')
x_train['education'] = x_train['education'].astype('category')
x_train['default'] = x_train['default'].astype('boolean')
x_train['housing'] = x_train['housing'].astype('boolean')
x_train['loan'] = x_train['loan'].astype('boolean')
x_train['contact'] = x_train['contact'].astype('category')
x_train['poutcome'] = x_train['poutcome'].astype('category')
x_train['day_of_week'] = x_train['day_of_week'].astype('category')
x_train['month'] = x_train['month'].astype('category')

x_val['job'] = x_val['job'].astype('category')
x_val['marital'] = x_val['marital'].astype('category')
x_val['education'] = x_val['education'].astype('category')
x_val['default'] = x_val['default'].astype('boolean')
x_val['housing'] = x_val['housing'].astype('boolean')
x_val['loan'] = x_val['loan'].astype('boolean')
x_val['contact'] = x_val['contact'].astype('category')
x_val['poutcome'] = x_val['poutcome'].astype('category')
x_val['day_of_week'] = x_val['day_of_week'].astype('category')
x_val['month'] = x_val['month'].astype('category')


dmatrix_train = xgboost.DMatrix(x_train, label=y_train['target'], enable_categorical=True)
dmatrix_val = xgboost.DMatrix(x_val, label=y_val['target'], enable_categorical=True)


xgb_params = {}

xgb_params['objective'] = 'reg:logistic'
xgb_params['eval_metric'] = "logloss"

xgb_params_lst = list(xgb_params.items())


num_rounds = 1000
evallist = [(dmatrix_val, 'val')]
model = xgboost.train(xgb_params_lst, dmatrix_train, num_rounds, evals=evallist, early_stopping_rounds = 50)


model.save_model('model.json')

storage_client = storage.Client()
BUCKET_NAME = <your bucket name>
bucket = storage_client.get_bucket(BUCKET_NAME)
blob = bucket.blob(<path on bucket to store the model at> + 'model.json')
blob.upload_from_filename('model.json')
