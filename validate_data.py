import pandas as pd
from scipy.stats import zscore

#print data in file for easier visualization
def	print_data_in_file(file):
	file_rows = file.head()
	file_columns = list(file.columns())

	#information about the file such as column names, number of non-null values,
	#to check for data cleanliness
	file_info = file.info()
	print ("\n<<<<<<<<<<<<<<<Rows>>>>>>>>>>>>>>>\n\n", file_rows)
	print ("\n<<<<<<<<<<<<<<<Columns>>>>>>>>>>>>>>>\n\n", file_columns)
	print ("\n<<<<<<<<<<<<<<<Info>>>>>>>>>>>>>>>\n\n", file_info)

#check the data in file
def	check_data_in_file(file_name, data_file):
	#check for null values in cells
	if data_file.isnull().sum().sum() != 0:
		print(f"Error: null value in {file_name}")
		return False
	#check for duplicates
	if data_file.duplicated().sum().sum() != 0:
		print(f"Error: duplicate found in {file_name}")
		return False
	#check for invalid booleans
	if not data_file['hasYard'].isin([0,1]).any():
		print(f"Invalid boolean value {data_file['hasYard']} in {file_name} 'hasYard'")
		return False
	if not data_file['hasPool'].isin([0,1]).any():
		print(f"Invalid boolean value in {file_name} 'hasPool'")
		return False
	if not data_file['isNewBuilt'].isin([0,1]).any():
		print(f"Invalid boolean value in {file_name} 'isNewBuilt'")
		return False
	if not data_file['hasStormProtector'].isin([0,1]).any():
		print(f"Invalid boolean value in {file_name} 'hasStormProtector'")
		return False
	if not data_file['hasStorageRoom'].isin([0,1]).any():
		print(f"Invalid boolean value in {file_name} 'hasStorageRoom'")
		return False
	return True

def look_for_outliers(file):
	z_scores = file.apply(zscore)
	threshold = 2
	outliers = (z_scores.abs() > threshold).any(axis=1)
	if outliers.any():
		print(f"Outliers were found on {file[outliers]}")
		return False
	return True

#load files and validate data
def load_and_validate_files(train_file, test_file):
	#check if the files have the correct extension
	if not train_file.endswith('.csv') or not test_file.endswith('csv'):
		print ("Error: wrong file extension. CSV file required.")
		return False 

	#get the content of the files
	try:
		train = pd.read_csv(train_file)
		test = pd.read_csv(test_file)
	except:
		print ("Error: could not read files.")
		return False

	train = train.iloc[:, 1:]
	test = test.iloc[:, 1:]
	#check data for duplicates or null values
	if not check_data_in_file(train_file, train) or \
		not check_data_in_file(test_file ,test):
		return False

	#check for outliers
	if not look_for_outliers(train) or not look_for_outliers(test):
		return False

	#check columns for inconsistent data between two files
	if list(train.columns) != list(test.columns) + ['price']:
		print ("Error: inconsistent data. Files contain different columns")
		return False
	print ("Data is clean")
	return True