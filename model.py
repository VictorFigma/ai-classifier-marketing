import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Read the data
user_data = pd.read_csv("train/user_train.csv", delimiter=';')
session_data = pd.read_csv("train/session_train.csv", delimiter=',')

# Merge the data
user_session_data = pd.merge(user_data, session_data, on='user_id', how='outer')
user_summary = user_session_data.groupby('user_id').agg({
  'age': 'first',
  'abandoned_cart': 'last',
  'user_category': 'first',
  'marketing_target': 'first',
  'timestamp': 'nunique',
  'device_type': 'last',
  'browser': 'last',
  'operating_system': 'last',
  'ip_address': 'nunique',
  'country': 'first',
  'search_query': 'nunique',
  'page_views': 'sum',
  'session_duration': 'mean'
})

# Define features and target variable
X = user_summary.drop('marketing_target', axis=1)
y = user_summary['marketing_target']

# Define categorical and numerical columns
categorical_cols = ['user_category', 'device_type', 'browser', 'operating_system', 'country']
numerical_cols = ['age', 'abandoned_cart', 'timestamp', 'ip_address', 'search_query', 'page_views', 'session_duration']

# Preprocessing
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
clf.fit(X_train, y_train)

# Load test data
user_data_test = pd.read_csv("test/user_test.csv", delimiter=';')
session_data_test = pd.read_csv("test/session_test.csv", delimiter=',')

user_session_data = pd.merge(user_data_test, session_data_test, on='user_id', how='outer')
user_summary = user_session_data.groupby('user_id').agg({
  'age': 'first',
  'abandoned_cart': 'last',
  'user_category': 'first',
  'test_id': 'first',
  'timestamp': 'nunique',
  'device_type': 'last',
  'browser': 'last',
  'operating_system': 'last',
  'ip_address': 'nunique',
  'country': 'first',
  'search_query': 'nunique',
  'page_views': 'sum',
  'session_duration': 'mean'
})

# Predict
predictions = clf.predict(user_summary.drop('test_id', axis=1))

results_dict = {"target": {}}
for test_id, prediction in zip(user_summary['test_id'], predictions):
    results_dict["target"][str(test_id)] = int(prediction)

with open('predictions/predictions.json', 'w') as outfile:
    json.dump(results_dict, outfile)