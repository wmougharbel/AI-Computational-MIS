from linear_regression import linear_regression_model, calculate_model_metrics, add_predicted_price_to_csv

def main():
	train_file = 'data/train.csv'
	test_file = 'data/test.csv'
	predicted_price = linear_regression_model(train_file, test_file)
	add_predicted_price_to_csv(test_file, predicted_price)

if __name__ == '__main__':
	main()