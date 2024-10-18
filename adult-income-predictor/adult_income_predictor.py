import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing               #to be used for one hot encoding
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer     #imputing (replacing null values)
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import math
import plotly.express as px
import matplotlib
import matplotlib.pyplot as         plt
import seaborn as sns

def main():

	f_LogisticRegression()
 
def f_LogisticRegression():

	raw_df = pd.read_csv('adults_train.csv', sep=',')
	raw_df = raw_df.map(lambda x: x.strip() if isinstance(x, str) else x)   #remove leading space from categorical columns


#	print(raw_df.query('CapGain > 0 & CapLoss > 0'))   	# no such rows exist
#	capGain_df = raw_df[raw_df['CapGain'] != 0] 
#	capLoss_df = raw_df[raw_df['CapLoss'] != 0] 
#	cap_df = pd.concat([capGain_df,capLoss_df], axis=0)
#	cap_df.loc[cap_df['CapLoss'] > 0, 'CapGain'] = 'Minor'
#	print(cap_df)
#	print(cap_df.shape)
#	print(cap_df.info())

	raw_df['Cap'] = np.where(raw_df['CapLoss'] > 0, raw_df['CapGain']-raw_df['CapLoss'], raw_df['CapGain'])
	raw_df = raw_df.drop('CapGain', axis=1)
	raw_df = raw_df.drop('CapLoss', axis=1)

	non_zero_mask = raw_df['Cap'] != 0
	non_zero_values = raw_df.loc[non_zero_mask, 'Cap'].values.reshape(-1, 1)
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_values = scaler.fit_transform(non_zero_values)
	raw_df.loc[non_zero_mask, 'Cap'] = scaled_values

#	print(raw_df.info())
#	print(raw_df.head())
	
#	print(raw_df.loc[:, raw_df.isnull().any()].columns)

#	print(raw_df.Workclass.unique())
#	print(raw_df.Education.unique())
#	print(raw_df.MaritalStatus.unique())
#	print(raw_df.Occupation.unique())
#	print(raw_df.Relationship.unique())
#	print(raw_df.Race.unique())
#	print(raw_df.Sex.unique())
#	print(raw_df.NativeCountry.unique())

#	print('Workclass values - ') 
#	print(raw_df.Workclass.value_counts())
#	print(raw_df.Education.value_counts())
#	print(raw_df.MaritalStatus.value_counts())
#	print('Occupation values - ') 
#	print(raw_df.Occupation.value_counts())
#	print(raw_df.Relationship.value_counts())
#	print(raw_df.Race.value_counts())
#	print(raw_df.Sex.value_counts())
#	print('Native country values - ') 
#	print(raw_df.NativeCountry.value_counts())

#	raw_df["Workclass"] = raw_df["Workclass"].astype('string')
#	raw_df["Education"] = raw_df["Education"].astype('string')
#	raw_df["MaritalStatus"] = raw_df["MaritalStatus"].astype('string')
#	raw_df["Occupation"] = raw_df["Occupation"].astype('string')
#	raw_df["Relationship"] = raw_df["Relationship"].astype('string')
#	raw_df["Race"] = raw_df["Race"].astype('string')
#	raw_df["Sex"] = raw_df["Sex"].astype('string')
#	raw_df["NativeCountry"] = raw_df["NativeCountry"].astype('string')

#	raw_df.drop(raw_df[raw_df.Workclass == '?'].index,inplace=True)
#	raw_df.drop(raw_df[raw_df.Occupation == '?'].index, inplace=True)
#	raw_df.drop(raw_df[raw_df.NativeCountry == '?'].index, inplace=True)


#	print(raw_df.loc[:, raw_df.isnull().any()].columns)

#	print('Workclass values - ') 
#	print(raw_df.Workclass.value_counts())
#	print('Occupation values - ') 
#	print(raw_df.Occupation.value_counts())
#	print('Native country values - ') 
#	print(raw_df.NativeCountry.value_counts())

	numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
	categorical_cols = raw_df.select_dtypes(include='object').columns.tolist()
	categorical_cols.remove('Income')

#	print(numeric_cols)
#	print(categorical_cols)

	scale_cols = ['Age','HourPerWeek']
	scaler = MinMaxScaler()
	scaler.fit(raw_df[scale_cols])    
	raw_df[scale_cols] = scaler.transform(raw_df[scale_cols]) 
#	print(raw_df.loc[:,'Age'])
#	print(raw_df.loc[:,'HourPerWeek'])

#	raw_df['capGainLoss'] = raw_df.apply(capGainLoss, axis=1)	
#	scale_cols = ['capGainLoss']
#	scaler = MinMaxScaler(feature_range=(-1,1))
#	scaler.fit(raw_df[scale_cols])    
#	raw_df[scale_cols] = scaler.transform(raw_df[scale_cols]) 
#	print(raw_df.loc[raw_df['capGainLoss'] < 0])
#	print(raw_df.loc[raw_df['capGainLoss'] > 0])
#	print(raw_df.loc[raw_df['capGainLoss'] == 0])

	encoder = OneHotEncoder(sparse_output=False, handle_unknown = 'ignore')
	one_hot_encoded = encoder.fit_transform(raw_df[categorical_cols])
	one_hot_df = pd.DataFrame(one_hot_encoded,columns=encoder.get_feature_names_out(categorical_cols))

	train_df = pd.concat([raw_df,one_hot_df],axis=1)
	train_df = train_df.drop(categorical_cols,axis=1)
	train_df = train_df.drop('Income', axis = 1)

#	print(encoder.categories_)
#	encoded_cols = list(encoder.get_feature_names_out(categorical_cols)) #create list of col names to be used in df
#	print('Encoded cols - ')
#	print(encoded_cols)
#	raw_df[encoded_cols] = encoder.transform(raw_df[categorical_cols]) 
   
	target_df = raw_df['Income']
#	print('Target DF -')
#	print(target_df)

#	print(target_df.shape)
#	print(train_df.shape)
#	print(train_df.info())
#	print(list(train_df))
#	print(train_df.head())

	model = LogisticRegression(solver='liblinear')
	model.fit(train_df, target_df)

	train_preds = model.predict(train_df)

	print(train_preds.shape)
#	print(train_preds.head())
	print(train_preds)
#	print(target_df)

	print(accuracy_score(target_df,train_preds))
	train_probs = model.predict_proba(train_df)      # instead of Yes/No prediction this will give probability of Yes and No
	print(train_probs)
	cm = confusion_matrix(target_df,train_preds,normalize='true') # Create a matrix of true positive/negative & false +/-
	print(cm)

	disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#		display_labels=target_df.target_names)
#	disp.plot(cmap=plt.cm.Blues)
#	plt.title('Confusion Matrix')
#	plt.show()

	return

def capGainLoss(row):

	if (row['CapGain'] > 0):
		return row['CapGain']
	else:
		if (row['CapLoss'] > 0):
			return row['CapLoss']*-1
		else:
			return 0

if __name__== "__main__":
   main()
