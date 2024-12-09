import pandas as pd
import numpy as np
import math

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from sklearn.model_selection import RandomizedSearchCV


def main():

	""" 

This model is for predicting health insurance premium, using Linear Regression from sklearn library

Data file (insurance.csv) is from Kagle competition
Number of records - 1339.

Features :
	age -				numeric
	sex - 			categoric (male / female)
	bmi - 			numeric
	children - 		numeric
	smoker - 		categoric (yes / no)
	region - 		categoric (4 unique vaalues)

Target :
	charges - numeric

Understanding Data
	1. There are no null values in any of the column
	2. age ranges from 18 to 64
	3. bmi ranges from 15.96 to 53.13
	4. children ranges from 0 to 5
	5. there are no outliers in the data

Feature Engineering
	1. scale age, bmi, children between 0 to 1 using minmax scaler from sklearn
	2. change value of sex column to numeric, 0 = male, 1 = female
	3. change value of smoker column to numeric, 0 = yes, 1 = no
	4. do one hot encoding for region 

	"""

	input_df = pd.read_csv('insurance.csv')

	input_df.loc[input_df["sex"] == 'male', "sex"] = 0
	input_df.loc[input_df["sex"] == 'female', "sex"] = 1
	
	input_df.loc[input_df["smoker"] == 'yes', "smoker"] = 0
	input_df.loc[input_df["smoker"] == 'no', "smoker"] = 1

	numeric_cols = ['age','bmi','children']
	scaler = MinMaxScaler()
	scaled_data = scaler.fit_transform(input_df[numeric_cols])	# returns array
	input_df[numeric_cols] = scaled_data

	encoder = OneHotEncoder(sparse_output=False)
	cat_cols = ['region']
	one_hot_encoded = encoder.fit_transform(input_df[cat_cols])	 # returns array
	encoded_df = pd.DataFrame(one_hot_encoded,columns=encoder.get_feature_names_out(cat_cols))	# convert array into data frame

	final_df = pd.concat([input_df,encoded_df], axis=1)	# concat the original data + ohe columns
	final_df.drop(['region'],inplace=True,axis=1)			# and drop the original column that has now been expanded into encoded columns
	target_df = final_df['charges']
	final_df.drop(['charges'],inplace=True,axis=1)

#	split the df into train & test
	
	x_train, x_test, y_train, y_test = train_test_split(final_df,target_df)
#	print(x_train.shape)
#	print(x_test.shape)
#	print(y_train.shape)
#	print(y_test.shape)
	

	regr = LinearRegression()
	regr.fit(x_train,y_train)

	predicted_charges = regr.predict(x_test)				#returns array as checked using isinstance(predicted_charges, pd.DataFrame)
	print(len(predicted_charges))
	print(len(y_test))
#	print(isinstance(y_train, pd.DataFrame))
#	print(isinstance(y_train, pd.DataFrame))

	mse = 0
	for actual_charge, predicted_charge in zip(y_test, predicted_charges):
#		print(actual_charge, predicted_charge)
		mse = mse + ((actual_charge - predicted_charge)**2)
	rmse = math.sqrt(mse)
	print(f'Root Mean Square Error is {rmse}')
	print(f'Model accuracy is {r2_score(y_test, predicted_charges)*100}')

#	Tunning hyperparameters to optimize the model
	print(regr.get_params())
	param_space = {'copy_X': [True,False], 
                  'fit_intercept': [True,False], 
                  'n_jobs': [1,5,10,15,None], 
                  'positive': [True,False]
                 }

	random_search = RandomizedSearchCV(regr, param_space, n_iter=15, cv=5)
	random_search.fit(x_train, y_train)
	predicted_charges = regr.predict(x_test)

	mse = 0
	for actual_charge, predicted_charge in zip(y_test, predicted_charges):
#		print(actual_charge, predicted_charge)
		mse = mse + ((actual_charge - predicted_charge)**2)
	rmse = math.sqrt(mse)
	print(f'After hyperparameter tunning - Root Mean Square Error is {rmse}')
	print(f'After hyperparameter tunning - Model accuracy is {r2_score(y_test, predicted_charges)*100}')


if __name__== "__main__":
   main()

