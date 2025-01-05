import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from validate_data import load_and_validate_files

def get_correlation_and_select_features(train):
	#calculate correlation
	correlation = train.corr()
	features = []
	target = ['price']
	price_correlation = correlation['price'].sort_values(ascending=False)

	#select variables with higher correlation than 0
	for key, value in price_correlation.items():
		if value > 0 and key != 'price':
			features.append(key)
	return features, target

def linear_regression_model(train_file, test_file):
	#validate files
	if not load_and_validate_files(train_file, test_file):
		return False
	train = pd.read_csv(train_file)
	test = pd.read_csv(test_file)
 
	#store features and target in respective variables
	X_train = train[get_correlation_and_select_features(train)[0]]
	Y_train = train[get_correlation_and_select_features(train)[1]]
	X_train_split, X_validation_split, Y_train_split, Y_validation_split = \
		train_test_split(X_train, Y_train)
	
	#initialize linear regression model and fit it in the dataset
	model = LinearRegression()
	model.fit(X_train_split, Y_train_split)

	#predict price for validation in train
	Y_validation_pred = model.predict(X_validation_split)
	Y_train_pred = model.predict(X_train_split)

	#predict price for the test dataset X_test = X_train
	X_test = test[get_correlation_and_select_features(train)[0]]
	predicted_price = model.predict(X_test)
 
	mse_train = mean_squared_error(Y_train_split, Y_train_pred)
	rmse_train = root_mean_squared_error(Y_train_split, Y_train_pred)
	print(f"Training MSE: {mse_train}")
	print(f"Training RMSE: {rmse_train}")
 
	return Y_validation_split, Y_validation_pred, predicted_price

def add_predicted_price_to_csv(file, predicted_price):
	file_df = pd.read_csv(file)
	file_df['predicted_price'] = predicted_price
	file_df.to_csv('data/test_prediction.csv')

def calculate_model_metrics(Y_validation_split, Y_validation_pred):
	mse = mean_squared_error(Y_validation_split, Y_validation_pred)
	rmse = root_mean_squared_error(Y_validation_split, Y_validation_pred)
	r2 = r2_score(Y_validation_split, Y_validation_pred)
	return mse, rmse, r2 
