from linear_regression import linear_regression_model, calculate_model_metrics, add_predicted_price_to_csv

def main():
	train_file = 'data/train.csv'
	test_file = 'data/test.csv'
	Y_validation_split, Y_validation_pred, predicted_price = linear_regression_model(train_file, test_file)
	add_predicted_price_to_csv(test_file, predicted_price)
	mse, rmse, r2 = calculate_model_metrics(Y_validation_pred, Y_validation_split)
	print (f"MSE: {mse}\nRMSE: {rmse}\nR2: {r2}")

if __name__ == '__main__':
	main()