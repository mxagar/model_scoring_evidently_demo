"""This example shows how the Evidently library
can be used to monitor ML models.

The Bicycle Demand Data from UCI is used.

Everything is defined in this main.py file.
Basically, we create HTML dashboards with Evidently
and save them to `./static`. Then, a FastAPI web app
is instantiated to which we pass all the content in that
`./static` folder.

These are the sections in the file:

- Dataset is loaded and split: reference and current
- Features are differentiated: categorical, numerical, etc.
- Scikit-Learn model is defined and trained ono reference split
- A column mapping is instantiated for evidently
- Training outputs are passed to evidently
- Model performance is computed for the current split: complete and weekly
- Data drift is computed for weeks 1 & 2 using another column mapping

"""
import pandas as pd
#import numpy as np
#import requests
#import zipfile
#import io

from datetime import datetime
from sklearn import datasets, ensemble

from evidently.dashboard import Dashboard
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import (DataDriftTab,
                                      NumTargetDriftTab,
                                      RegressionPerformanceTab)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

#content = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip").content
#with zipfile.ZipFile(io.BytesIO(content)) as arc:
#    raw_data = pd.read_csv(arc.open("hour.csv"), 
#                            header=0, 
#                            sep=',', 
#                            parse_dates=['dteday'], 
#                            index_col='dteday')
filename = "./data/hour.csv"
raw_data = pd.read_csv(filename, 
                       header=0, 
                       sep=',', 
                       parse_dates=['dteday'], 
                       index_col='dteday')
#raw_data.head()

## Regression training

target = 'cnt'
prediction = 'prediction'
numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr', 'weekday']
categorical_features = ['season', 'holiday', 'workingday']

# Split: refrence dataset vs. current dataset; January 2011 and February 2011
reference = raw_data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00']
current = raw_data.loc['2011-01-29 00:00:00':'2011-02-28 23:00:00']

reference.head()

regressor = ensemble.RandomForestRegressor(random_state = 0, 
                                           n_estimators = 50)

regressor.fit(reference[numerical_features + categorical_features], 
              reference[target])

ref_prediction = regressor.predict(reference[numerical_features + categorical_features])
current_prediction = regressor.predict(current[numerical_features + categorical_features])

reference['prediction'] = ref_prediction
current['prediction'] = current_prediction

## Model Performance

column_mapping = ColumnMapping()

column_mapping.target = target
column_mapping.prediction = prediction
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features

regression_perfomance_dashboard = Dashboard(tabs=[RegressionPerformanceTab()])
regression_perfomance_dashboard.calculate(reference, None, column_mapping=column_mapping)

# Show/save to HTML
#regression_perfomance_dashboard.show()
regression_perfomance_dashboard.save("./static/index.html")

##  Week 1

regression_perfomance_dashboard.calculate(reference,
                                          current.loc['2011-01-29 00:00:00':'2011-02-07 23:00:00'], 
                                          column_mapping=column_mapping)

#regression_perfomance_dashboard.show()
regression_perfomance_dashboard.save("./static/regression_performance_after_week1.html")

target_drift_dashboard = Dashboard(tabs=[NumTargetDriftTab()])
target_drift_dashboard.calculate(reference,
                                 current.loc['2011-01-29 00:00:00':'2011-02-07 23:00:00'], 
                                 column_mapping=column_mapping)

#target_drift_dashboard.show()
target_drift_dashboard.save("./static/target_drift_after_week1.html")

## Week 2

regression_perfomance_dashboard.calculate(reference,
                                          current.loc['2011-02-07 00:00:00':'2011-02-14 23:00:00'], 
                                          column_mapping=column_mapping)

#regression_perfomance_dashboard.show()
regression_perfomance_dashboard.save("./static/regression_performance_after_week2.html")

target_drift_dashboard.calculate(reference,
                                 current.loc['2011-02-07 00:00:00':'2011-02-14 23:00:00'], 
                                 column_mapping=column_mapping)

#target_drift_dashboard.show()
target_drift_dashboard.save("./static/target_drift_after_week2.html")

## Week 3

regression_perfomance_dashboard.calculate(reference,
                                          current.loc['2011-02-15 00:00:00':'2011-02-21 23:00:00'], 
                                          column_mapping=column_mapping)

#regression_perfomance_dashboard.show()
regression_perfomance_dashboard.save("./static/regression_performance_after_week3.html")

target_drift_dashboard.calculate(reference,
                                 current.loc['2011-02-15 00:00:00':'2011-02-21 23:00:00'], 
                                 column_mapping=column_mapping)

#target_drift_dashboard.show()
target_drift_dashboard.save("./static/target_drift_after_week3.html")

## Data Drift Week 1

column_mapping = ColumnMapping()

column_mapping.numerical_features = numerical_features

data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
data_drift_dashboard.calculate(reference,
                               current.loc['2011-01-29 00:00:00':'2011-02-07 23:00:00'], 
                               column_mapping=column_mapping)

#data_drift_dashboard.show()
data_drift_dashboard.save("./static/data_drift_dashboard_after_week1.html")

## Data Drift Week 2

column_mapping = ColumnMapping()
column_mapping.numerical_features = numerical_features
data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
data_drift_dashboard.calculate(reference,
                               current.loc['2011-02-07 00:00:00':'2011-02-14 23:00:00'],
                               column_mapping=column_mapping)
data_drift_dashboard.save("./static/data_drift_dashboard_after_week2.html")


## Web App based on FastAPI
app = FastAPI()
app.mount("/", StaticFiles(directory="static", html=True), name="static")
