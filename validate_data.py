import pandas as pd

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

	#check data for duplicates or null values
	if not check_data_in_file(train_file, train) or \
		not check_data_in_file(test_file ,test):
		return False

	#check columns for inconsistent data
	if list(train.columns) != list(test.columns) + ['price']:
		print ("Error: inconsistent data. Files contain different columns")
		return False
	print ("Data is clean")
	return True