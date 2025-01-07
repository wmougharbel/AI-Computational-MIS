import pandas as pd
from linear_regression import linear_regression_model

def add_predicted_price_to_csv(file, predicted_price):
	file_df = pd.read_csv(file)
	file_df['predicted_price'] = predicted_price
	file_df.to_csv('data/test_prediction.csv', index=False)

def main():
	train_file = 'data/train.csv'
	test_file = 'data/test.csv'
	predicted_price = linear_regression_model(train_file, test_file)
	add_predicted_price_to_csv(test_file, predicted_price)

if __name__ == '__main__':
	main()