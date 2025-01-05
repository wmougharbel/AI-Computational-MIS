from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

def calculate_model_metrics(Y_validation_split, Y_validation_pred, Y_train_split, Y_train_pred, model,X_train, Y_train):
	#model performance with validation subset
	test_mse = mean_squared_error(Y_validation_split, Y_validation_pred)
	test_rmse = root_mean_squared_error(Y_validation_split, Y_validation_pred)
	test_r2 = r2_score(Y_validation_split, Y_validation_pred)
 
	#model performance with trained subset
	train_mse = mean_squared_error(Y_train_split, Y_train_pred)
	train_rmse = root_mean_squared_error(Y_train_split, Y_train_pred)
	train_r2 = r2_score(Y_train_split, Y_train_pred)

	#check for overfitting with cross validation. The mean should be as close to 1 as possible
	#overfitting occurs when the model memorizes the dataset instead of generalizing
	#It would work with one dataset but fail with others
	scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='r2')
	scores_mean = scores.mean()
 
	#1% of the mean price can be taken as a threshold
	rmse_threshold = Y_train.mean().values[0] * 0.01
	mse_threshold = rmse_threshold ** 2

	print(f"{YELLOW}\nModel Performance Metrics:\n{RESET}")

	#Mean of prices in train data
	print(f"Mean of Prices: {Y_train.mean().values[0]}")

	# Training metrics
	print(f"{GREEN if train_mse < mse_threshold else RED}Training MSE: {train_mse:.3f} ({'Good' if train_mse < mse_threshold else 'Needs improvement'}){RESET}")
	print(f"{GREEN if train_rmse < rmse_threshold else RED}Training RMSE: {train_rmse:.3f} ({'Good' if train_rmse < rmse_threshold else 'High'}){RESET}")
	print(f"{GREEN if train_r2 > 0.95 else (RED if train_r2 <= 0.80 else RESET)}Training R^2: {train_r2:.3f} ({'Excellent' if train_r2 > 0.95 else 'Acceptable' if train_r2 > 0.80 else 'Poor'}){RESET}")
	# Validation metrics
	print(f"{GREEN if test_mse < mse_threshold else RED}Training MSE: {train_mse:.3f} ({'Good' if test_mse < mse_threshold else 'Needs improvement'}){RESET}")
	print(f"{GREEN if test_rmse < rmse_threshold else RED}Training RMSE: {train_rmse:.3f} ({'Good' if test_rmse < rmse_threshold else 'High'}){RESET}")
	print(f"{GREEN if test_r2 > 0.95 else (RED if test_r2 <= 0.80 else RESET)}Training R^2: {train_r2:.3f} ({'Excellent' if test_r2 > 0.95 else 'Acceptable' if train_r2 > 0.80 else 'Poor'}){RESET}")

	# Cross-validation
	print(f"{YELLOW}\nCross-Validation Metrics:{RESET}")
	print(f"Cross-Validation R^2 Scores: {scores}")
	print(f"Mean R^2: {scores_mean:.3f} ({'Excellent' if scores_mean > 0.95 else 'Acceptable' if scores_mean > 0.80 else 'Poor'})")

	# Overfitting assessment
	if abs(train_r2 - test_r2) > 0.1:
		print("\nWarning: Potential overfitting detected! Large gap between training and validation R^2.")
	else:
		print("\nModel generalizes well.")
