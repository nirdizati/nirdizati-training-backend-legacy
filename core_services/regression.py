import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

import cPickle
import pandas as pd
import numpy as np

from os.path import isfile


def linear(fileName, prefix, encoding, cluster):
	if isfile('core_results/linearregression_'+ encoding  + '_' + fileName + '_' + str(prefix) + '.csv'):
		return None
	
	train_data, test_data, original_test_data = prep_data(fileName, prefix, encoding)
   	lm = LinearRegression(fit_intercept=True)
   	y = train_data['remainingTime']
   	train_data = train_data.drop('remainingTime', 1)
   	lm.fit(train_data, y)
   	with open('core_predictionmodels/linearregression_'+ encoding + '_' + fileName + '_' + str(prefix) +'.pkl', 'wb') as fid:
   		cPickle.dump(lm, fid)

   	original_test_data['prediction'] = lm.predict(test_data)
   	original_test_data.to_csv('core_results/linearregression_' + encoding + '_' + fileName + '_' + str(prefix) + '.csv', sep=',', mode='w+', index=False)
    
def randomforestregression(fileName, prefix, encoding, cluster):
	if isfile('core_results/randomforestregression_'+ encoding  + '_' + fileName + '_' + str(prefix) + '.csv'):
		return None
	
	train_data, test_data, original_test_data = prep_data(fileName, prefix, encoding)
	rf = RandomForestRegressor(n_estimators=50, n_jobs=8, verbose=1)
	y = train_data['remainingTime']
	train_data = train_data.drop('remainingTime', 1)
	rf.fit(train_data, y)
	with open('core_predictionmodels/randomforestregression_'+ encoding + '_' + fileName + '_' + str(prefix) +'.pkl', 'wb') as fid:
		cPickle.dump(rf, fid)

	original_test_data['prediction'] = rf.predict(test_data)

	original_test_data.to_csv('core_results/randomforestregression_' + encoding + '_' + fileName + '_' + str(prefix) + '.csv',sep=',',mode='w+', index=False)

#def xgboost(fileName, prefix, encoding, cluster):
def split_data(data):
    cases = data['Id'].unique()
    import random
    random.shuffle(cases)

    cases_train_point = int(len(cases) * 0.8)

    train_cases = cases[:cases_train_point]

    ids = []
    for i in range(0, len(data)):
        ids.append(data['Id'][i] in train_cases)

    train_data = data[ids]
    test_data = data[np.invert(ids)]
    return train_data, test_data

def prep_data(fileName, prefix, encoding):
    df = pd.read_csv(filepath_or_buffer='core_encodedFiles/' + encoding + '_' + fileName + '_' + str(prefix) +  '.csv', header=0)
    train_data, test_data = split_data(df)

    train_data = train_data.drop('Id', 1)
    original_test_data = test_data
    test_data = test_data.drop('Id', 1)

    test_data = test_data.drop('remainingTime', 1)

    return train_data, test_data, original_test_data


    # boolean_filename_prefix, frequency_filename_prefix, complex_index_filename_prefix, simple_index_filename_prefix, index_latest_payload
